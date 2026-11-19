# Glider DVL Analysis documentation

## Description

OOI Pioneer-New England Shelf glider DVL quality control, validation, and analysis of NES shelf
currents.

## Package modules

| Module | Description |
| --- | --- |
| `binning` | Bin the ADCP velocities and science data of each profile by depth |
| `bottom_track` | Remove glider motion from DVL water velocities using bottom tracking, and check the inversion with it |
| `corrections` | Correct the DVL time offset from the glider clock and the DVL vertical velocity for the sound speed |
| `currents` | Depth-averaged currents from the glider's dead-reckoned and GPS positions |
| `flight` | Glider velocity through the water from the gliderflight steady-state flight model |
| `inversion` | Absolute velocity profiles from the DVL and flight model by least squares, referenced to the depth-averaged current or bottom tracking |
| `mapping` | Cartopy maps of glider tracks and mooring locations |
| `merge` | Merge IOOS Glider DAC datasets with glider DVL/ADCP data from OOINet, and build a glider dataset through the dives, climbs, and surfacings |
| `objective_mapping` | Objective mapping (optimal interpolation) of glider sections |
| `profiles` | Split glider ADCP datasets into individual profiles, label the dives, climbs, and surfacings, and link the ADCP and science profiles |
| `qc` | Quality control of ADCP data based on TRDI recommendations |
| `transects` | Split the deployment into transects along the survey box edges |
| `utils` | Shared methods across modules |

## Commands

The Makefile contains the central entry points for common tasks related to this project. Run
`make` to list them.
