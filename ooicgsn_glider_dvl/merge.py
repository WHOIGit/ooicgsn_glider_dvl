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
        if gdac[var].dims[0] == "profile":
            profile_vars.append(var)
        elif "instrument" in gdac[var].attrs:
            if var == "platform_meta":
                continue
            elif "ctd" in gdac[var].attrs["instrument"]:
                ctd_vars.append(var)
            elif "oxy" in gdac[var].attrs["instrument"]:
                oxy_vars.append(var)
            elif "par" in gdac[var].attrs["instrument"]:
                par_vars.append(var)
            elif "flbbcd" in gdac[var].attrs["instrument"]:
                flbbcd_vars.append(var)
            else:
                pass
        elif "source_sensor" in gdac[var].attrs and "instrument" not in gdac[var].attrs:
            glider_vars.append(var)
        else:
            pass

    # Create a dictionary of each possible instrument with
    # associated GDAC dataset variables
    sensor_vars = {
        "ctd": ctd_vars,
        "oxy": oxy_vars,
        "flbbcd": flbbcd_vars,
        "par": par_vars,
        "glider": glider_vars,
        "profile": profile_vars,
    }

    return sensor_vars


def split_data(gdac: xr.Dataset, sensor_vars: list) -> xr.Dataset:
    """Splits off the given science variables"""
    sensor_data = gdac[sensor_vars].dropna(dim="time")
    return sensor_data


def interp_to_time(data: xr.Dataset, time: np.ndarray) -> dict[str, xr.Variable]:
    """Linearly interpolate the 1-D data variables of a dataset to a new time base

    Times outside of the time range of the data are filled with NaN rather
    than extrapolated. Variables that are neither numeric nor datetimes are
    skipped.

    Parameters
    ----------
    data: xarray.Dataset
        Dataset with a "time" dimension and no missing values
    time: numpy.ndarray
        The datetime64 time base to interpolate to

    Returns
    -------
    interp: dict[str, xarray.Variable]
        The interpolated variables, keyed by variable name, with their attributes
    """
    data = data.sortby("time")
    x = data["time"].values.astype("datetime64[ns]").astype("int64")
    xi = np.asarray(time).astype("datetime64[ns]").astype("int64")

    interp = {}
    for v in data.data_vars:
        values = data[v].values
        if np.issubdtype(values.dtype, np.datetime64):
            ns = values.astype("datetime64[ns]").astype("int64").astype(float)
            yi = np.interp(xi, x, ns, left=np.nan, right=np.nan)
            values = np.full(yi.shape, np.datetime64("NaT"), dtype="datetime64[ns]")
            valid = ~np.isnan(yi)
            values[valid] = yi[valid].astype("int64").astype("datetime64[ns]")
        elif np.issubdtype(values.dtype, np.number):
            values = np.interp(xi, x, values, left=np.nan, right=np.nan)
        else:
            continue
        interp[v] = xr.Variable("time", values, data[v].attrs)

    return interp


def merge_datasets(dvl: xr.Dataset, gdac: xr.Dataset) -> xr.Dataset:
    """Merge the DVL and GDAC glider datasets

    Select and interpolate the science and glider sensor
    data from the OOI GDAC glider dataset to the associated
    OOI DVL glider dataset. First, the science and glider
    sensor data are separately split into separate datasets.
    Then they are interpolated to the DVL timebase. Finally
    they are merged into the DVL datasets. DVL times outside
    of a sensor's time range are filled with NaN.

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

    # Next, get the data for each variable set and interpolate
    # it to the dvl time base, with the source sensor in the name
    merged_vars = {}
    for sensor, variables in sensor_vars.items():
        variables = [v for v in variables if v not in gdac.coords]
        if sensor == "profile" or not variables:
            continue
        sensor_data = split_data(gdac, variables)
        interp_sensor = interp_to_time(sensor_data, dvl["time"].values)
        merged_vars.update({f"{sensor}_{v}": var for v, var in interp_sensor.items()})

    # Add all of the interpolated variables to the dvl dataset at once
    return dvl.assign(merged_vars)


def add_waypoints(dvl: xr.Dataset, waypoints: pd.DataFrame) -> xr.Dataset:
    """Merge the waypoint data into the glider DVL dataset

    Take the waypoint target lat/lon ("c_wpt_lat/lon") from the
    glider engineering dataset and map it to the glider DVL
    dataset. Each waypoint applies from the time it is set until
    the next waypoint is set; DVL times before the first waypoint
    are filled with NaN.

    Parameters
    ----------
    dvl: xarray.Dataset
        The OOI glider DVL dataset downloaded from OOI
    waypoints: pandas.DataFrame
        The associated glider engineering data downloaded from OOI
        as a csv file and loaded into a pandas DataFrame. Only the
        "time", "c_wpt_lat", and "c_wpt_lon" columns are used.

    Returns
    -------
    dvl: xarray.Dataset
        The OOI glider DVL dataset with the waypoint lat/lons
        mapped and broadcasted to the DVL timebase
    """

    # First, clean up and select just the relevant waypoints
    waypoints = waypoints[["time", "c_wpt_lat", "c_wpt_lon"]].dropna()
    waypoints = waypoints.assign(time=pd.to_datetime(waypoints["time"])).sort_values("time")

    # Keep only the times where either the latitude or longitude changes
    d_wpt = waypoints[["c_wpt_lat", "c_wpt_lon"]].diff()
    mask = (d_wpt["c_wpt_lat"] == 0) & (d_wpt["c_wpt_lon"] == 0)
    waypoints = waypoints[~mask]

    # Find the most recent waypoint set at or before each DVL time
    wpt_time = waypoints["time"].values.astype("datetime64[ns]")
    dvl_time = dvl["time"].values.astype("datetime64[ns]")
    idx = np.searchsorted(wpt_time, dvl_time, side="right") - 1
    valid = idx >= 0

    wpt_lat = np.full(dvl_time.shape, np.nan)
    wpt_lon = np.full(dvl_time.shape, np.nan)
    wpt_lat[valid] = waypoints["c_wpt_lat"].values[idx[valid]]
    wpt_lon[valid] = waypoints["c_wpt_lon"].values[idx[valid]]

    # Add the waypoints to the merged dataset
    dvl["waypoint_lat"] = (["time"], wpt_lat)
    dvl["waypoint_lat"].attrs = {
        "long_name": "Glider Waypoint Latitude",
        "comment": ("The target waypoint latitude for the glider."),
        "units": "degrees",
    }

    dvl["waypoint_lon"] = (["time"], wpt_lon)
    dvl["waypoint_lon"].attrs = {
        "long_name": "Glider Waypoint Longitude",
        "comment": ("The target waypoint longitude for the glider."),
        "units": "degrees",
    }

    return dvl


def downcast(ds: xr.Dataset, dims: tuple[str, ...] = ("bin",)) -> xr.Dataset:
    """Convert float64 variables along the given dimensions to float32

    The DVL measurements do not need double precision, so storing the
    (time, bin) variables as float32 halves the memory and file size.
    Coordinates and the 1-D (time) variables, such as lat/lon, are kept
    at their original precision.
    """
    for v in ds.data_vars:
        if ds[v].dtype == np.float64 and any(d in ds[v].dims for d in dims):
            ds[v] = ds[v].astype(np.float32)
    return ds


def netcdf_encoding(ds: xr.Dataset, complevel: int = 4) -> dict[str, dict]:
    """Build a to_netcdf encoding that compresses all of the data variables"""
    return {v: {"zlib": True, "complevel": complevel} for v in ds.data_vars}
