import numpy as np
from scipy.linalg import toeplitz, cho_factor, cho_solve
import matplotlib.pyplot as plt
import finufft

# 0) always centre signals
def demean(x):
    x = np.asarray(x, float).ravel()
    return x - x.mean()

# --- 1) sample autocov & Toeplitz C (use biased for smoother C) ---
def sample_autocov(y: np.ndarray, maxlag: int | None = None, unbiased: bool = False) -> np.ndarray:
    y = demean(y)
    N = len(y)
    if maxlag is None:
        maxlag = N - 1
    r = np.empty(maxlag + 1, dtype=float)
    for k in range(maxlag + 1):
        x  = y[:N - k]
        xk = y[k:]
        denom = (N - k) if unbiased else N  # keep unbiased=False for smoother Toeplitz
        r[k] = float(np.dot(x, xk) / denom)
    return r

def covariance_from_y_toeplitz(y, maxlag=None, unbiased=False, ridge=0.0):
    r = sample_autocov(y, maxlag=maxlag, unbiased=unbiased)
    C = toeplitz(r[:len(y)])
    if ridge > 0.0:
        C = C + ridge * np.eye(len(y))
    return C, r

# --- 2) PSD from the SAME autocov (use this PSD everywhere) ---
def pxx_from_autocov(r: np.ndarray, fs: float, eps_floor: float = 1e-12):
    r = np.asarray(r, float).ravel()
    r_even = np.concatenate([r, r[-2:0:-1]]) if len(r) > 1 else r.copy()
    L = len(r_even)
    freqs = np.fft.rfftfreq(L, d=1.0/fs)
    S = np.fft.rfft(r_even).real * (1.0 / fs)   # two-sided (on rfft grid)
    Pxx = S.copy()
    if Pxx.size > 2:
        Pxx[1:-1] *= 2.0                        # one-sided
    # tiny floor (same as you do in NUFFT path)
    med = np.median(Pxx[Pxx > 0]) if np.any(Pxx > 0) else 1.0
    Pxx = np.maximum(Pxx, eps_floor * med)
    return freqs, Pxx

# --- 3) FFT-domain matched filter: use the SAME PSD and weights ---
def matched_filter_snr(lc: np.ndarray, template_tr: np.ndarray, pxx_one_sided: np.ndarray) -> float:
    lc = demean(lc); template_tr = demean(template_tr)
    n = len(lc)
    X = np.fft.rfft(lc)
    H = np.fft.rfft(template_tr)

    weights = np.ones_like(pxx_one_sided)
    if n % 2 == 0:
        weights[1:-1] = 2.0
    else:
        weights[1:] = 2.0

    S = np.maximum(pxx_one_sided, 1e-20)
    num = np.real(np.sum((X * np.conj(H)) * weights / S))
    den = np.sqrt(np.real(np.sum((np.abs(H) ** 2) * weights / S)))
    return num / den

# --- 4) NUFFT-domain: same grid, same PSD, same weights ---
def matched_filter_snr_nufft(t_lc, lc, t_template, template_tr, f_grid, pxx_one_sided_on_grid):
    lc = demean(lc); template_tr = demean(template_tr)
    t   = np.asarray(t_lc, float).ravel()
    t_h = np.asarray(t_template, float).ravel()
    w   = np.asarray(2.0 * np.pi * f_grid, float).ravel()
    cx  = np.asarray(lc, complex).ravel()
    ch  = np.asarray(template_tr, complex).ravel()

    Xk = finufft.nufft1d3(t,   cx, w, isign=1, eps=1e-12)
    Hk = finufft.nufft1d3(t_h, ch, w, isign=1, eps=1e-12)

    S = np.maximum(pxx_one_sided_on_grid, 1e-20)
    weights = np.ones_like(S)
    if len(S) > 2:
        weights[1:-1] = 2.0

    num = np.real(np.sum((Xk * np.conj(Hk)) * weights / S))
    den = np.sqrt(np.real(np.sum((np.abs(Hk) ** 2) * weights / S)))
    return num / den

# --- 5) Time-domain GLS (C from y), unchanged but demean inputs ---
def matched_filter_time(y: np.ndarray, t: np.ndarray, C: np.ndarray) -> float:
    y = demean(y); t = demean(t)
    cf = cho_factor(C, lower=True, check_finite=False)
    Cy = cho_solve(cf, y, check_finite=False)
    Ct = cho_solve(cf, t, check_finite=False)
    return float((t @ Cy) / np.sqrt(t @ Ct))

# usage
#snr_nufft = matched_filter_snr_nufft(t_mask, lc_mask, t_mask, template_mask, f_fft, pxx_for_grid)
#snr_fft = matched_filter_snr(lc, template_tr, pxx_for_grid)
