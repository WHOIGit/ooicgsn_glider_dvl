# Glider DVL Analysis
OOI Pioneer-New England Shelf glider DVL quality control, validation, and analyis of NES shelf currents

##### Authors
Andrew Reed: [profile](https://github.com/reedan88), areed@whoi.edu
Brayden Mikulski: [link to gitHub profile], brayden.mikulski@whoi.edu

## Overview
The Ocean Observatories Initiative (OOI) is a NSF-funded project for long-term infrastructure for ocean observations. The Coastal & Global Scale (CGSN) nodes group based at WHOI builds, deploys, and maintains moored and mobile autonomous platforms equipped with sensors that measure physical, meteorological, biogeochemical, and biological properties. CGSN is also responsible for quality controlling the data via the use of automated algorithms, validation against discrete ship-based samples, and cross-comparison between co-located instruments.

This project will use autonomous glider DVL (doppler velocity log) water velocity data from the OOI Pioneer-New England Shelf array. The DVL data will be used to derive estimates of the along-shelf currents for comparison with geostrophic calculations. The first part of the project will involve quality-controlling the DVL data using vendor supplied algorithms as well as developing their own methods. The quality-controlled DVL data will then be validated against moored velocity data when the gliders are near the moorings. This validated dataset will then be ready to construct cross-shelf transects of water velocities and compare against calculated geostrophic velocities.


## Project Organization

```
├── LICENSE
├── Makefile             <- Convenience commands (`make create_environment`, `make lint`, ...)
├── README.md
├── environment.yml      <- Conda environment; installs this package in editable mode
├── pyproject.toml       <- Package metadata and ruff configuration
│
├── data                 <- Not tracked by git (see .gitignore)
│   ├── external         <- Third-party data, e.g. mooring_locations.csv
│   ├── raw              <- Data downloaded from OOINet / IOOS Glider DAC
│   └── processed        <- Merged and quality-controlled datasets
│
├── docs                 <- mkdocs project; see www.mkdocs.org
│
├── notebooks            <- Analysis notebooks, numbered in run order
│   ├── 1-data-access-and-download.ipynb
│   ├── 2-Process-and-merge-datasets.ipynb
│   ├── 3-Data-analysis.ipynb
│   └── 4-Remove-bottom-track.ipynb
│
├── references           <- Papers and other explanatory materials
│
├── results
│   └── figures          <- Generated figures
│
└── ooicgsn_glider_dvl   <- Source code for this project
    ├── __init__.py
    ├── mapping.py              <- Cartopy maps of glider tracks and mooring locations
    ├── merge.py                <- Merge IOOS Glider DAC datasets with glider DVL/ADCP data from OOINet
    ├── objective_mapping.py    <- Objective mapping (optimal interpolation) of glider sections
    ├── profiles.py             <- Split glider ADCP datasets into individual profiles
    ├── qc.py                   <- Quality control of ADCP data based on TRDI recommendations
    └── utils.py                <- Shared methods across modules
```

## Getting Started

```
make create_environment
conda activate ooicgsn_glider_dvl
```

Modules are then importable from the notebooks, e.g. `from ooicgsn_glider_dvl import qc`.

--------

