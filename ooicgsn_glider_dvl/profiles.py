"""
@author Andrew Reed
@brief Provides methods for splitting glider ADCP datasets into separate profiles
"""

import numpy as np
import xarray as xr

from ooicgsn_glider_dvl.utils import unix_time


def identify_profiles(ds: xr.Dataset) -> list[np.ndarray]:
    """Identify the start of each profile dive.

    The glider ADCP is only on during the dive portion
    of each glider profile. We identify where the change
    in depth between each measurement (~4 seconds) is greater
    than 2 meters, indicating the start of a new dive.

    Parameters
    ----------
    ds: xarray.Dataset
        A dataset of the glider ADCP data with the parameter
        'depth_from_pressure'

    Returns
    -------
    inflection_idx: list[numpy.array[int]]
        The indices of the data points in each profile dive
    """
    # Take the absolute difference in depth
    dz = np.abs(np.diff(ds["depth_from_pressure"].values))

    # Get the time id
    time_ = unix_time(ds["time"].values)

    # Identify where the difference is greater than 2 meters
    # and get the associated times with those inflection points
    inflection_times = np.concatenate([time_[:1], time_[:-1][dz > 2], time_[-1:]])

    # Each profile spans (inflection_times[n], inflection_times[n+1]], so the
    # profile of each point is the inflection interval it falls in
    interval = np.searchsorted(inflection_times, time_, side="left") - 1
    valid = np.flatnonzero(interval >= 0)
    if valid.size == 0:
        return []

    # Group the indices by profile, dropping empty intervals
    order = np.argsort(interval[valid], kind="stable")
    sorted_interval = interval[valid][order]
    starts = np.flatnonzero(np.diff(sorted_interval)) + 1
    inflection_idx = np.split(valid[order], starts)

    return inflection_idx


def get_profile_ids(ds: xr.Dataset) -> np.ndarray[int]:
    """Create profile ids for the glider ADCP dataset

     Parameters
    ----------
    ds: xarray.Dataset
        A dataset of the glider ADCP data with the parameter
        'depth_from_pressure'

    Returns
    -------
    profiles: numpy.array[int]
        An array the length of the input dataset with the
        profile id added based on the inflection indices
    """

    # First, get the inflection indices
    profile_idx = identify_profiles(ds)

    # Next, number each profile sequentially
    profiles = np.zeros(ds["time"].shape, dtype=np.int32)
    for n, idx in enumerate(profile_idx):
        profiles[idx] = n

    return profiles


def add_profiles(ds: xr.Dataset) -> xr.Dataset:
    """Add a sequential id to each glider profile dive

    The glider ADCP is only on during the dive portion
    of each glider profile. We identify where the change
    in depth between each measurement (~4 seconds) is greater
    than 2 meters, indicating the start of a new dive. Each
    sampling point is then matched with the start and stop times
    of an individual profile, and the profile id is added to
    the dataset.

    Parameters
    ----------
    ds: xarray.Dataset
        A dataset of the glider ADCP data with the parameter
        'depth_from_pressure'

    Returns
    -------
    ds: xarray.Dataset
        The input dataset of the glider ADCP data with each
        time point matched with its profile dive id

    """
    # Get the profiles
    profiles = get_profile_ids(ds)

    # Add to the dataset
    ds["profile_id"] = (["time"], profiles)
    ds["profile_id"].attrs = {
        "comment": (
            "The id of each profile in the dataset counting sequentially from the first profile."
        ),
        "long_name": "Profile ID",
        "ancillary_variables": "depth_from_pressure",
    }

    return ds
