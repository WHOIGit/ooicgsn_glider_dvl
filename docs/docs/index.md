# Glider DVL Analysis documentation

## Description

OOI Pioneer-New England Shelf glider DVL quality control, validation, and analysis of NES shelf
currents.

## Package modules

| Module | Description |
| --- | --- |
| `bottom_track` | Remove glider motion from DVL water velocities using bottom tracking |
| `currents` | Depth-averaged currents from the glider's dead-reckoned and GPS positions |
| `mapping` | Cartopy maps of glider tracks and mooring locations |
| `merge` | Merge IOOS Glider DAC datasets with glider DVL/ADCP data from OOINet |
| `objective_mapping` | Objective mapping (optimal interpolation) of glider sections |
| `profiles` | Split glider ADCP datasets into individual profiles |
| `qc` | Quality control of ADCP data based on TRDI recommendations |
| `transects` | Split the deployment into transects along the survey box edges |
| `utils` | Shared methods across modules |
| `velocity_profiles` | Bin the ADCP velocities of each profile by depth |

## Commands

The Makefile contains the central entry points for common tasks related to this project. Run
`make` to list them.
