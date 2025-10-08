#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author Andrew Reed
@brief Provides methods for merging datasets from IOOS GDAC with GLider DVL/ADcP datasets from OOINet
"""

import numpy as np
import pandas as pd
import xarray as xr

def sensor_variables(gdac: xr.Dataset) -> dict[list]:
    """Match IOOS GDAC dataset variables to science sensor source
    
    Iterate throu an OOI IOOS GDAC glider dataset and split out
    the dataset variables into separate lists based on what sensor
    they originate from. This is a necessary preprocessing step before
    interpolation because the different sensor sample at different rates,
    such that the GDAC dataset is populated with NaNs.
    
    Parameters
    ----------
    gdac: xarray.Dataset
        The loaded OOI GDAC glider dataset with science data
        
    Returns
    -------
    sensor_vars: dict[lists]
        A dictionary with the GDAC dataset variables associated
        with specific science sensors or the glider sensors
    """

    # Initialize various lists to keep dataset variables in
    ctd_vars = []
    oxy_vars = []
    flbbcd_vars = []
    par_vars = []
    glider_vars = []
    profile_vars = []
    
    # Iterate through the dataset variables and look for the source
    # sensor attribute and match to appropriate list
    for var in gdac.variables:
        if gdac[var].dims[0] == 'profile':
            profile_vars.append(var)
        elif 'instrument' in gdac[var].attrs:
            if var == 'platform_meta':
                continue
            elif 'ctd' in gdac[var].attrs['instrument']:
                ctd_vars.append(var)
            elif 'oxy' in gdac[var].attrs['instrument']:
                oxy_vars.append(var)
            elif 'par' in gdac[var].attrs['instrument']:
                par_vars.append(var)
            elif 'flbbcd' in gdac[var].attrs['instrument']:
                flbbcd_vars.append(var)
            else:
                pass
        elif 'source_sensor' in gdac[var].attrs and 'instrument' not in gdac[var].attrs:
            glider_vars.append(var)
        else:
            pass

    # Create a dictionary of each possible instrument with
    # associated GDAC dataset variables
    sensor_vars = {
        'ctd': ctd_vars,
        'oxy': oxy_vars,
        'flbbcd': flbbcd_vars,
        'par': par_vars,
        'glider': glider_vars,
        'profile': profile_vars
    }

    return sensor_vars


def split_data(gdac: xr.Dataset, sensor_vars: list) -> xr.Dataset:
    """Splits off the given science variables"""
    sensor_data = gdac[sensor_vars].dropna(dim='time')
    return sensor_data


def merge_datasets(dvl: xr.Dataset, gdac: xr.Dataset) -> xr.Dataset:
    """Merge the DVL and GDAC glider datasets
    
    Select and interpolate the science and glider sensor
    data from the OOI GDAC glider dataset to the associated
    OOI DVL glider dataset. First, the science and glider 
    sensor data are separately split into separate datasets.
    Then they are interpolated to the DVL timebase. Finally
    they are merged into the DVL datasets.
    
    Parameters
    ----------
    dvl: xarray.Dataset
        The OOI glider DVL dataset downloaded from OOI
    gdac: xarray.Dataset
        The associated OOI glider data downloaded from the
        IOOS GDAC server
        
    Returns
    -------
    dvl: xarray.Dataset
        The glider DVL data with the science and glider
        data from the GDAC dataset interpolated to the DVL
        timebase and merged
    """
    # First, get the sensor variables
    sensor_vars = sensor_variables(gdac)

    # Next, get the data for each variable set, interpret
    # it to the dvl time base, and merge into the dvl dataset
    for sensor in sensor_vars:
        if sensor == 'profile':
            continue
        else:
            sensor_data = split_data(gdac, sensor_vars[sensor])
            # Now rename the variables to have the source sensor
            # in the name
            for v in sensor_data.variables:
                if v in sensor_data.coords:
                    pass
                else:
                    sensor_data = sensor_data.rename_vars({v: '_'.join((sensor, v))})
            
            # Now interpolate the sensor data to the dvl time base
            interp_sensor = sensor_data.interp_like(dvl['time'], method='linear', kwargs={'fill_value':'extrapolate'})
    
            # Check if depth is a coordinate, and if so, drop it
            if 'depth' in interp_sensor.coords:
                interp_sensor = interp_sensor.drop_vars('depth')
    
            # Merge the interpolated sensor data with the dvl data
            dvl = xr.merge([dvl, interp_sensor])

    return dvl


def add_waypoints(dvl: xr.Dataset, waypoints: pd.DataFrame) -> xr.Dataset:
    """Merge the waypoint data into the glider DVL dataset
    
    Take the waypoint target lat/lon ("c_wpt_lat/lon") from the
    glider engineering dataset and map it to the glider DVL 
    dataset.
    
    Parameters
    ----------
    dvl: xarray.Dataset
        The OOI glider DVL dataset downloaded from OOI
    waypoints: pandas.DataFrame
        The associated glider engineering data downloaded from OOI
        as a csv file and loaded into a pandas DataFrame

    Returns
    -------
    dvl: xarray.Dataset
        The OOI glider DVL dataset with the waypoint lat/lons 
        mapped and broadcasted to the DVL timebase
    """

    # First, clean up and select just the relevant waypoints
    waypoints = waypoints[['time','c_wpt_lat','c_wpt_lon']].dropna()
    waypoints['time'] = waypoints['time'].map(pd.to_datetime)

    # Take the difference and find where either the latitude or longitude changes
    d_wpt = waypoints.diff()
    mask = (d_wpt['c_wpt_lat'] == 0) & (d_wpt['c_wpt_lon'] == 0)
    waypoints = waypoints[~mask]

    # Now add waypoints tot he tie base of the DVL dataset
    # First, create empty arrays to hold the data
    wpt_lat = np.zeros(dvl['time'].shape)
    wpt_lon = np.zeros(dvl['time'].shape)
    wpt_time = dvl['time'].values

    # Iterate through the waypoints and find the appropriate times that they
    # apply to
    for n, (t, lat, lon) in enumerate(waypoints.itertuples(index=False)):
        if n == 0:
            # At the start of the deployment, only have first time
            idx, = np.where(wpt_time <= t)
            # Now add in the waypoint lat
            wpt_lat[idx] = lat
            wpt_lon[idx] = lon
        else:
            t0 = waypoints['time'].iloc[n-1]
            idx, = np.where((wpt_time > t0) & (wpt_time <= t))
            wpt_lat[idx] = lat
            wpt_lon[idx] = lon

    # Add the waypoints to the merged dataset
    dvl['waypoint_lat'] = (['time'], wpt_lat)
    dvl['waypoint_lat'].attrs = {
        'long_name': 'Glider Waypoint Latitude',
        'comment': ('The target waypoint latitude for the glider.'),
        'units': 'degrees'
        }
    
    dvl['waypoint_lon'] = (['time'], wpt_lon)
    dvl['waypoint_lon'].attrs = {
        'long_name': 'Glider Waypoint Longitude',
        'comment': ('The target waypoint longitude for the glider.'),
        'units': 'degrees'
    }

    return dvl
    