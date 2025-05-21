# Glider DVL Analysis

OOI Pioneer-New England Shelf glider DVL quality control, validation, and analyis of NES shelf currents

## Overview
The Ocean Observatories Initiative (OOI) is a NSF-funded project for long-term infrastructure for ocean observations. The Coastal & Global Scale (CGSN) nodes group based at WHOI builds, deploys, and maintains moored and mobile autonomous platforms equipped with sensors that measure physical, meteorological, biogeochemical, and biological properties. CGSN is also responsible for quality controlling the data via the use of automated algorithms, validation against discrete ship-based samples, and cross-comparison between co-located instruments.

This project will use autonomous glider DVL (doppler velocity log) water velocity data from the OOI Pioneer-New England Shelf array. The DVL data will be used to derive estimates of the along-shelf currents for comparison with geostrophic calculations. The first part of the project will involve quality-controlling the DVL data using vendor supplied algorithms as well as developing their own methods. The quality-controlled DVL data will then be validated against moored velocity data when the gliders are near the moorings. This validated dataset will then be ready to construct cross-shelf transects of water velocities and compare against calculated geostrophic velocities.


## Project Organization

```
├── LICENSE            <- Open-source license if one is chosen
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── docs               <- A default mkdocs project; see www.mkdocs.org for details
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── pyproject.toml     <- Project configuration file with package metadata for 
│                         glider_dvl_analysis and configuration for tools like black
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
├── setup.cfg          <- Configuration file for flake8
│
└── glider_dvl_analysis   <- Source code for use in this project.
    │
    ├── __init__.py             <- Makes glider_dvl_analysis a Python module
    │
    ├── config.py               <- Store useful variables and configuration
    │
    ├── dataset.py              <- Scripts to download or generate data
    │
    ├── features.py             <- Code to create features for modeling
    │
    ├── modeling                
    │   ├── __init__.py 
    │   ├── predict.py          <- Code to run model inference with trained models          
    │   └── train.py            <- Code to train models
    │
    └── plots.py                <- Code to create visualizations
```

--------

