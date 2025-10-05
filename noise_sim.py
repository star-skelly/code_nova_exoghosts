import numpy as np
from noise_generator import powerlaw_psd, simulate_from_psd
import matplotlib.pyplot as plt
NUM = 1000
N = 90*24*2
FS = 1/120
ALPHA = 2.0
VAR = 2.0
SEED = 0
OUT = "star_noise_cohort.npz"

rng = np.random.default_rng(SEED)
_, S = powerlaw_psd(N, FS, alpha=ALPHA, variance=VAR)
data = np.vstack([
    simulate_from_psd(S, N, FS, rng=np.random.default_rng(rng.integers(0, 2**63 - 1)))
    for _ in range(NUM)
])

np.savez_compressed(OUT, data=data, n_samples=N, fs=FS, alpha=ALPHA, variance=VAR, seed=SEED,
                    description="Spectral noise cohort generated via powerlaw PSD and simulate_from_psd")

