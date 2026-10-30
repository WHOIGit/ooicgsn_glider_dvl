# Getting started

## Set up the environment

```bash
make create_environment
conda activate ooicgsn_glider_dvl
```

This installs the `ooicgsn_glider_dvl` package in editable mode, so its modules can be imported
from the notebooks (e.g. `from ooicgsn_glider_dvl import qc`). Notebook 1 also needs the OOI
data-access packages `ooinet` and `ooi_data_explorations`, which are installed separately from
GitHub.

## Run the notebooks

The notebooks in `notebooks/` are numbered in run order. Data are written to `data/`, which is not
tracked by git.

1. **`1-data-access-and-download.ipynb`**: downloads the glider ADCP data (OOI JupyterHub kdata),
   glider engineering data (OOINet THREDDS), and glider science data (IOOS Glider DAC) for each
   Pioneer-NES glider deployment into `data/raw/`, and mooring locations into `data/external/`.
2. **`2-Process-and-merge-datasets.ipynb`**: merges the Glider DAC science data onto the DVL time
   base, splits the data into profiles, applies the ADCP quality-control tests, and saves the
   merged dataset to `data/processed/`.
3. **`3-Data-analysis.ipynb`**: maps the glider tracks, cleans the velocities, and compares them
   against the moored ADCP data.
4. **`4-Remove-bottom-track.ipynb`**: examines the bottom-track velocities and removes the glider
   motion from the DVL water velocities.

Figures are saved to `results/figures/`.
