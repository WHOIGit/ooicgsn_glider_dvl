"""
@author Andrew Reed
@brief Provides methods for removing glider motion from DVL water velocities using bottom tracking
"""

import xarray as xr


def remove_bottom_track(glider: xr.Dataset, bottom_track: xr.Dataset) -> xr.Dataset:
    """Remove the bottom track from the glider DVL water velocities.

    Parameters
    ----------
    glider: xarray.Dataset
        Merged glider dataset with the variables 'eastward_seawater_velocity'
        and 'northward_seawater_velocity' (m s-1)
    bottom_track: xarray.Dataset
        Glider ADCP bottom-track dataset with the variables
        'bt_eastward_velocity' and 'bt_northward_velocity' (mm s-1)

    Returns
    -------
    glider: xarray.Dataset
        The glider dataset with the added variables 'corrected_eastward_velocity'
        and 'corrected_northward_velocity'
    """
    # Adjust the eastward velocities
    east_bottom_track = bottom_track["bt_eastward_velocity"] / 1000
    eastward_velocity = glider["eastward_seawater_velocity"] - east_bottom_track

    # Adjust the northward velocities
    north_bottom_track = bottom_track["bt_northward_velocity"] / 1000
    northward_velocity = glider["northward_seawater_velocity"] - north_bottom_track

    # Add the results into the glider dataset
    glider["corrected_eastward_velocity"] = eastward_velocity
    glider["corrected_eastward_velocity"].attrs = {
        "standard_name": "eastward_seawater_velocity",
        "long_name": "Motion-Corrected Eastward Seawater Velocity",
        "units": "m s-1",
        "comment": (
            "Eastward seawater velocity component in Earth coordinates. The motion of the "
            "glider has been removed by subtracting the eastward velocity measured by the "
            "bottom tracking, when available"
        ),
        "ancillary_variables": "eastward_seawater_velocity bt_eastward_velocity",
    }

    glider["corrected_northward_velocity"] = northward_velocity
    glider["corrected_northward_velocity"].attrs = {
        "standard_name": "northward_seawater_velocity",
        "long_name": "Motion-Corrected Northward Seawater Velocity",
        "units": "m s-1",
        "comment": (
            "Northward seawater velocity component in Earth coordinates. The motion of the "
            "glider has been removed by subtracting the northward velocity measured by the "
            "bottom tracking, when available"
        ),
        "ancillary_variables": "northward_seawater_velocity bt_northward_velocity",
    }

    return glider
