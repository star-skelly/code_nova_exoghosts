# EXOGHOSTS: Hunting for exo-ghosts in light curves

Short project summary for the GitHub page.

## What data we use
- TESS light curves (processed/normalized) stored as pickles under `TESS_data/processed_data/` with filenames like `<TID>_normalized.p`.
- Each target (TID) uses 3 sectors of TESS observations (≈ 90 days total per TID) from year 6.
- Sampling cadence: 2 minutes per sample.
- Scripts operate in “samples” internally and convert to days/hours for visualization:
  - Period [days] = samples / (30 · 24)
  - Duration [hours] = samples / 30

### `.p` file structure
Pickle files contain at least:
- `cadence` (np.ndarray): integer sample indices (uniform 2‑min cadence). Missing samples may be present (see below).
- `processed_flux` (np.ndarray): normalized flux (around 1.0). In examples we use `lc = processed_flux - 1` to center at 0.

Notes:
- Gaps/missing data are represented with NaNs in `processed_flux` (and may appear implicitly via masked cadences). Our PSD estimator is gap‑aware and ignores non‑finite samples.

## What we do
- Estimate the stellar noise power spectrum with a gap‑aware smoothed one‑sided periodogram (`smoothed_periodogram_with_gaps`).
- Run a NUFFT-based matched‑filter transit search over a period/duration grid with two noise models:
  - Baseline white‑noise (flat PSD: `2·var/fs`)
  - Adaptive learned stellar model (smoothed PSD from the data)
- Visualize detection scores as heatmaps (period in days, duration in hours) and show phase‑folded light curves at selected parameters.

Data scale & coverage:
- Period grid spans multi‑day periods; duration grid spans hours. Defaults shown in `tess_transit_search_example.py` (tunable).

## Key components
- `tess_transit_search_example.py` — Loads a TESS light curve, builds white/adaptive PSDs, runs the grid search, saves NPZ for the dashboard, and plots reference figures.
- `noise_generator.py` — Utilities: transit templates, PSD estimators (including gap‑aware smoothing), phase fold + detrend helpers.
- `nufft_detector.py` — Matched‑filter SNR using the NUFFT and a provided one‑sided PSD.
- `dashboard.py` — Plotly Dash app showing side‑by‑side heatmaps (white vs adaptive) and phase‑folded light curves with the true‑parameter cross.

## Quick start
1) Generate the dashboard data NPZ from a TESS target (change `tid` in the script as needed):
```bash
python code_nova_exoghosts/tess_transit_search_example.py
```

2) Launch the dashboard:
```bash
python code_nova_exoghosts/dashboard.py
```

## Dependencies
- Python 3.10+
- numpy, matplotlib, plotly, dash

Optional: If you use different environments, ensure these packages are installed (e.g., `pip install numpy matplotlib plotly dash`).
