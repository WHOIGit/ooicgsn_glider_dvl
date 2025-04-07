#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author Andrew Reed
@brief Provides methods for splitting glider ADCP datasets into separate profiles
"""

import numpy as np
import xarray as xr


def unix_time(dt, unit='s'):
    """Convert numpy datetime64 object to unix time"""
    return (dt - np.datetime64('1970-01-01T00:00:00')) / np.timedelta64(1, unit)


def identify_profiles(ds: xr.Dataset) -> np.ndarray:
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
    inflection_idx: numpy.array[int]
        The indices indicating the start of a new profile dive
    """
    # Take the absolute different in pressure
    dz_dt = np.abs(ds['depth_from_pressure'].diff(dim='time'))

    # Get the time id
    dt = ds['time'].values
    time_ = unix_time(dt)
    
    # Identify where the difference is greater than 2 meters
    # and get the associated times with those inflection points
    idx = np.flatnonzero(dz_dt > 2)
    inflection_times = time_[idx]
    inflection_times = np.insert(
        inflection_times,
        [0, len(inflection_times)],
        [time_[0], time_[-1]])
    
    # Use the time range to get the indices for each profile
    inflection_idx = []
    for idx in range(len(inflection_times)-1):
        pstart = inflection_times[idx]
        pend = inflection_times[idx+1]
        profile_idx = np.flatnonzero(
            np.logical_and(
                time_ > pstart,
                time_ <= pend))
        if len(profile_idx) == 0:
            continue
        inflection_idx.append(profile_idx)

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
    
    # Next, iterate through the 
    profiles = np.zeros(ds['time'].shape)
    for n, idx in enumerate(profile_idx):
        profiles[idx] = n

    profiles = profiles.astype('int')
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
    ds['profile_id'] = (['time'], profiles)
    ds['profile_id'].attrs = {
        'comment': ('The id of each profile in the dataset '
                    'counting sequentially from the first profile.'),
        'long_name': 'Profile ID',
        'ancillary_variables': 'depth_from_pressure',
    }

    return ds