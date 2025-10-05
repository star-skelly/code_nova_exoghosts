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
