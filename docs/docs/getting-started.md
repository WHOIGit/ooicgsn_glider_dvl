# Getting started

## Set up the environment

The environment is managed with [uv](https://docs.astral.sh/uv/). Install uv, then from the
repository root:

```bash
make create_environment
```

This runs `uv sync`, which creates a `.venv` with the dependencies locked in `uv.lock` and the
`ooicgsn_glider_dvl` package installed in editable mode, so its modules can be imported from the
notebooks (e.g. `from ooicgsn_glider_dvl import qc`). It also registers the `.venv` as the Jupyter
kernel `Python (ooicgsn_glider_dvl)`, which the notebooks use. Run other commands in the
environment with `uv run`, e.g. `uv run jupyter lab`.

Notebooks 1 and 3.1 also need the OOI data-access packages `ooinet` and `ooi_data_explorations`.
Install them with `make ooi` (`uv sync --group ooi`). Loading data with `load_kdata` only works on
the OOI JupyterHub.

## Run the notebooks

The notebooks in `notebooks/` are numbered in run order. Data are written to `data/`, which is not
tracked by git.

1. **`1-data-access-and-download.ipynb`**: downloads the glider ADCP data (OOI JupyterHub kdata),
   glider engineering data (OOINet THREDDS), and glider science data (IOOS Glider DAC) for each
   Pioneer-NES glider deployment into `data/raw/`, and mooring locations into `data/external/`.
2. **`2-Process-and-merge-datasets.ipynb`**: merges the Glider DAC science data onto the DVL time
   base, splits the data into profiles, applies the ADCP quality-control tests, adds the glider
   waypoints and the depth-averaged current, and saves the merged dataset to `data/processed/`.
3. **`3-Data-analysis.ipynb`**: maps the glider tracks, cleans the velocities, and compares them
   against the moored ADCP data. **`3.1-alternative-data-analysis.ipynb`** splits the deployment
   into transects along the survey box edges and bins the velocities into depth profiles.
4. **`4-Remove-bottom-track.ipynb`**: examines the bottom-track velocities and removes the glider
   motion from the DVL water velocities.

Figures are saved to `results/figures/`.
