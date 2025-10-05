# Processed TESS lightcurve `.p` files

This repository stores processed TESS lightcurve datasets as Python pickle files (`*.p`).
This README explains where those files live, what they contain, and gives short examples for
loading, inspecting, visualizing, and aligning the data.

## Location / filename

- Default location used by the scripts in this repo: `~/TESS/data/`
- Files are named `<tic_id>.p` (for example `123456789.p`).

## File contents / data format

Each `.p` file is a pickle of a single tuple with eight elements, created by
the downloader script. The tuple layout is:

```
(lc_data, processed_lc_data, quality_data, time_data,
 cam_data, ccd_data, centroid_xy_data, pos_xy_corr)
```

Where each element is a dictionary keyed by sector number (integers). For a given sector `s`:

- `lc_data[s]` : SAP flux values (numpy array) with bad-quality cadences removed.
- `processed_lc_data[s]` : PDCSAP flux values (numpy array) with bad cadences removed — this is
  the primary processed flux used for plotting/analysis in the repo.
- `quality_data[s]` : quality mask array for the original cadences (0/1 style).
- `time_data[s]` : saved as `cadenceno` (integer cadence numbers) in the current scripts.
  Note: these are NOT wall-clock times (MJD/BJD) unless you updated the downloader to save
  `lc_sap.time` explicitly.
- `cam_data[s]` and `ccd_data[s]` : integers giving the camera and CCD for that sector.
- `centroid_xy_data[s]` : `[mom_centr1_array, mom_centr2_array]` arrays (centroid positions).
- `pos_xy_corr[s]` : `[pos_corr1_array, pos_corr2_array]` arrays (position corrections recorded by Lightkurve).

All per-sector arrays (flux, time, centroid, etc.) have the same length for a given sector after
the script's quality masking.

## Loading an example file

Python example to open and inspect a file:

```python
import pickle
from pathlib import Path

p = Path('~/TESS/data/123456789.p').expanduser()
with p.open('rb') as f:
    data = pickle.load(f)

# Unpack (same order as described above)
lc_data, proc_data, quality_data, time_data, cam_data, ccd_data, centroid_xy_data, pos_xy_corr = data

print('Sectors:', sorted(lc_data.keys()))
sector = sorted(lc_data.keys())[0]
print('Sector', sector, 'points:', len(proc_data[sector]))
```

Notes:
- Only unpickle files you trust — `pickle` can execute arbitrary code.

## Quick visualization

Here's a minimal example to plot processed flux for a single sector:

```python
import matplotlib.pyplot as plt
import numpy as np

sector = list(proc_data.keys())[0]
flux = np.asarray(proc_data[sector])
cad = np.asarray(time_data[sector])

plt.figure(figsize=(10,3))
plt.plot(cad, flux, '.', markersize=3)
plt.xlabel('cadence number')
plt.ylabel('PDCSAP flux')
plt.title(f'TIC {p.stem} sector {sector}')
plt.grid(True)
plt.show()
```

If you want real timestamps (MJD/BJD) instead of `cadenceno`, re-run the download step and
save `lc_sap.time` in the pickle — the current `download_lc.py` implementation stores cadence numbers.

## Combining & aligning sectors

The repository includes helper functions (in `preprocessing_TESSdata.py`) for combining
and aligning processed flux across sectors. The alignment routine attempts to find
overlaps between sectors and compute a robust additive (and optionally multiplicative)
correction so that fluxes are continuous across sector boundaries.


Load `.p` into `data` and call the `summarize(data)` helper to create the working `data_dict`.


Refer to the docstrings in `preprocessing_TESSdata.py` for parameter details.



