import numpy as np
import sys
import matplotlib.pyplot as plt

def boxtransit(times, period, dur, t0, alpha=1):
    """
    Generate a transit signal time-series with box function evaluated at given times.

    Args:
    times :  Array of time points at which to evaluate the transit time-series
    period : Period of the transit
    dur : Duration of the transit.
    t0 : Epoch.
    alpha : Transit depth. Default is 1.
    """

    return np.piecewise(times, [((times-t0+(dur/2))%period) > dur, ((times-t0+(dur/2))%period) <= dur], [0, 1])*(-alpha)

def simulate_from_psd(psd_values: np.ndarray, n_samples: int, fs: float, rng: np.random.Generator | None = None) -> np.ndarray:
	"""Simulate real Gaussian time series with a target one-sided PSD on the rFFT grid.
	"""
	if rng is None:
		rng = np.random.default_rng()

	psd_values = np.asarray(psd_values, dtype=float)
	assert psd_values.shape == (n_samples // 2 + 1,)

	# rFFT frequency vector defines the coefficient count
	freqs = np.fft.rfftfreq(n_samples, d=1.0 / fs)
	X = np.zeros_like(freqs, dtype=np.complex128)

	# Interior bins: complex Gaussian with scaling for desired PSD
	if n_samples % 2 == 0:
		interior = slice(1, -1)
	else:
		interior = slice(1, None)

	if X[interior].size > 0:
		a = rng.normal(size=X[interior].shape)
		b = rng.normal(size=X[interior].shape)
		scale = np.sqrt(psd_values[interior] * n_samples * fs / 4.0)
		X[interior] = (a + 1j * b) * scale

	# Enforce zero-mean and real signal constraints at DC and Nyquist
	X[0] = 0.0
	if n_samples % 2 == 0:
		X[-1] = 0.0

	x = np.fft.irfft(X, n=n_samples)
	x -= x.mean()
	return x


def powerlaw_psd(n_samples: int, fs: float, alpha: float = 2.0, variance: float = 1.0, f0: float | None = None) -> tuple[np.ndarray, np.ndarray]:
	"""Construct one-sided power-law PSD S(f) = A / (f0 + f)^alpha with target variance.
	Returns (freqs, S) on the rFFT grid with S[0]=0 for a zero-mean signal.
	"""
	freqs = np.fft.rfftfreq(n_samples, d=1.0 / fs)
	if f0 is None:
		f0 = fs / n_samples  # ~1/T as low-frequency regularizer

	shape = 1.0 / (f0 + freqs) ** alpha
	shape[0] = 0.0

	integral = (np.trapezoid if hasattr(np, "trapezoid") else np.trapz)(shape, freqs)
	if integral <= 0.0:
		raise ValueError("Non-positive PSD integral.")
	A = variance / integral
	S = A * shape
	return freqs, S


def simulate_cyclostationary(psd_values: np.ndarray, n_samples: int, fs: float, period: float, modulation_index: float = 0.5, phase: float = 0.0, rng: np.random.Generator | None = None) -> np.ndarray:
	"""Create cyclostationary Gaussian noise by periodic amplitude modulation of a base process.

	a(t) = 1 + m * sin(2π t / period + phase), normalized so E[a^2]=1.
	"""
	x_base = simulate_from_psd(psd_values, n_samples, fs, rng=rng)
	t = np.arange(n_samples) / fs
	a = 1.0 + modulation_index * np.sin(2.0 * np.pi * t / period + phase)
	a /= np.sqrt(1.0 + 0.5 * modulation_index ** 2)
	x = a * x_base
	x -= x.mean()
	return x


def periodogram_one_sided(x: np.ndarray, fs: float) -> tuple[np.ndarray, np.ndarray]:
	"""One-sided periodogram (power/Hz) with rFFT normalization."""
	n = len(x)
	X = np.fft.rfft(x)
	dt = 1.0 / fs
	freqs = np.fft.rfftfreq(n, d=dt)
	Pxx = (2.0 * dt / n) * (np.abs(X) ** 2)
	Pxx[0] = (dt / n) * (np.abs(X[0]) ** 2)
	if n % 2 == 0:
		Pxx[-1] = (dt / n) * (np.abs(X[-1]) ** 2)
	return freqs, Pxx


def smoothed_periodogram(
	x: np.ndarray,
	fs: float,
	kernel: str = "hann",
	width: int = 31,
	pad_mode: str = "reflect",
) -> tuple[np.ndarray, np.ndarray]:
	"""Smoothed one-sided periodogram via frequency-domain convolution.

	Args:
		x: time series (real)
		fs: sampling rate [Hz]
		kernel: 'hann' or 'boxcar'
		width: window length (bins), forced to odd
		pad_mode: padding for edges before convolution ('reflect' by default)

	Returns:
		freqs, Pxx_s: frequencies and smoothed power spectral density [power/Hz]
	"""
	# base periodogram
	freqs, Pxx = periodogram_one_sided(x, fs)
	# window (unit-sum) and convolution
	w = int(width) | 1
	if kernel == "boxcar":
		win = np.ones(w)
	else:
		# hann by default
		win = np.hanning(w)
	win = win / (win.sum() + 1e-12)
	pad = w // 2
	Px_pad = np.pad(Pxx, pad_width=pad, mode=pad_mode)
	Pxx_s = np.convolve(Px_pad, win, mode="valid")
	return freqs, Pxx_s

def detrend(t_vec: np.ndarray, y_vec: np.ndarray, deg: int = 2, mask: np.ndarray | None = None) -> np.ndarray:
    """Polynomial detrend in time; optionally fit on out-of-transit points only.

    Args:
        t_vec: time vector
        y_vec: flux vector
        deg: polynomial degree for trend
        mask: boolean mask of in-transit points (True=in-transit). If provided,
              the fit uses only out-of-transit samples (~mask).

    Returns:
        y_detrended: y_vec with polynomial trend removed.
    """
    t_vec = np.asarray(t_vec)
    y_vec = np.asarray(y_vec)
    if mask is not None and np.any(~mask):
        coefs = np.polyfit(t_vec[~mask], y_vec[~mask], deg=deg)
    else:
        coefs = np.polyfit(t_vec, y_vec, deg=deg)
    trend = np.polyval(coefs, t_vec)
    return y_vec - trend

def centered_phase_fold(t_vec: np.ndarray, period: float, dur: float, alpha_val: float) -> tuple[np.ndarray, np.ndarray]:
    """Compute centered phase and transit mask given a box transit template.

    Returns:
        phc: centered phase in [-0.5, 0.5)
        mask: boolean mask for in-transit samples
    """
    phase = (t_vec % period) / float(period)
    mask = boxtransit(t_vec, period=period, dur=dur, t0=0, alpha=alpha_val) != 0
    if np.any(mask):
        ang = np.angle(np.mean(np.exp(2j * np.pi * phase[mask])))
        center = (ang / (2 * np.pi)) % 1.0
    else:
        center = 0.0
    phc = (phase - center + 0.5) % 1.0 - 0.5
    return phc, mask

def _contiguous_true_runs(mask: np.ndarray) -> list[np.ndarray]:
    idx = np.where(mask)[0]
    if idx.size == 0:
        return []
    splits = np.where(np.diff(idx) > 1)[0] + 1
    runs = np.split(idx, splits)
    return runs

def average_periodogram_over_gaps(x: np.ndarray, fs: float, min_len: int = 256, nfft: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Average one-sided periodograms over contiguous finite segments (gap-aware).

    Uses rFFT with a common nfft grid for all segments to enable averaging.
    Each segment is mean-subtracted; the PSD normalization matches periodogram_one_sided.
    """
    x = np.asarray(x, float)
    ok = np.isfinite(x)
    runs = _contiguous_true_runs(ok)
    runs = [r for r in runs if r.size >= int(min_len)]
    if not runs:
        raise ValueError("No valid segments of sufficient length for PSD.")
    if nfft is None:
        nfft = max(r.size for r in runs)
    dt = 1.0 / fs
    freqs = np.fft.rfftfreq(nfft, d=dt)
    acc = np.zeros(freqs.shape, dtype=float)
    wsum = 0.0
    for r in runs:
        seg = x[r]
        seg = seg - np.mean(seg)
        X = np.fft.rfft(seg, n=nfft)
        Pxx = (2.0 * dt / nfft) * (np.abs(X) ** 2)
        Pxx[0] = (dt / nfft) * (np.abs(X[0]) ** 2)
        if nfft % 2 == 0:
            Pxx[-1] = (dt / nfft) * (np.abs(X[-1]) ** 2)
        w = float(seg.size)
        acc += w * Pxx
        wsum += w
    return freqs, acc / max(wsum, 1e-12)

def smoothed_periodogram_with_gaps(
    x: np.ndarray,
    fs: float,
    kernel: str = "hann",
    width: int = 31,
    pad_mode: str = "reflect",
    min_len: int = 256,
    nfft: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Gap-aware smoothed PSD: average segment periodograms, then smooth across frequency.
    """
    freqs, Pxx_avg = average_periodogram_over_gaps(x, fs, min_len=min_len, nfft=nfft)
    # Smooth the averaged PSD using the same windowing as smoothed_periodogram
    w = int(width) | 1
    if kernel == "boxcar":
        win = np.ones(w)
    else:
        win = np.hanning(w)
    win = win / (win.sum() + 1e-12)
    pad = w // 2
    Px_pad = np.pad(Pxx_avg, pad_width=pad, mode=pad_mode)
    Pxx_s = np.convolve(Px_pad, win, mode="valid")
    return freqs, Pxx_s

def _centered_phase_and_detrend(t_vec, y_vec, period, dur, alpha_val, deg=2):
    phase = (t_vec % period) / float(period)
    mask = boxtransit(t_vec, period=period, dur=dur, t0=0, alpha=alpha_val) != 0
    if np.any(mask):
        ang = np.angle(np.mean(np.exp(2j * np.pi * phase[mask])))
        center = (ang / (2 * np.pi)) % 1.0
    else:
        center = 0.0
    phc = (phase - center + 0.5) % 1.0 - 0.5
    idx_o = ~mask
    if np.any(idx_o):
        coefs = np.polyfit(phc[idx_o], y_vec[idx_o], deg=deg)
        trend = np.polyval(coefs, phc)
        y_dt = y_vec - trend
    else:
        y_dt = y_vec
    return phc, y_dt, mask