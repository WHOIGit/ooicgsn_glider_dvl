"""
@author Andrew Reed
@brief Provides methods for binning the glider ADCP velocities and science data of each glider
    profile by depth, to construct vertical profiles on a common depth grid
"""

import numpy as np
import xarray as xr

# ADCP velocities binned by default, on the time and bin (ADCP cell) dimensions
VELOCITY_VARIABLES = ("eastward_seawater_velocity", "northward_seawater_velocity")

# Science variables binned by default, on the time dimension, where they are in the dataset
SCIENCE_VARIABLES = (
    "ctd_temperature",
    "ctd_salinity",
    "ctd_conductivity",
    "ctd_density",
    "oxy_dissolved_oxygen",
    "oxy_oxygen_saturation",
    "flbbcd_chlorophyll",
    "flbbcd_backscatter",
    "flbbcd_CDOM",
    "par_PAR",
)

# Per-profile variables carried over to the binned profiles, where they are in the dataset:
# the transects and deviation flags (transects.py) and the depth-averaged current
# calculated from the glider's displacement (currents.py)
PROFILE_VARIABLES = (
    "transect_id",
    "transect_name",
    "transect_pass",
    "principal_direction",
    "profile_identifier",
    "along_distance",
    "cross_distance",
    "off_edge",
    "overshoot",
    "backtrack",
    "is_deviation",
    "deviation_segment",
    "depth_averaged_eastward_velocity",
    "depth_averaged_northward_velocity",
)


def bin_profiles(
    ds: xr.Dataset,
    bin_size: float = 2,
    max_depth: float | None = None,
    velocity_variables: tuple[str, ...] = VELOCITY_VARIABLES,
    science_variables: tuple[str, ...] | None = None,
    depth: str = "depth_from_pressure",
    qc_variable: str | None = "vendor_qc_flag",
    bad_flags: tuple[int, ...] = (4, 9),
    remove_bottom_hits: bool = True,
    lat: str = "lat",
    lon: str = "lon",
) -> xr.Dataset:
    """Construct vertical profiles of the ADCP velocities and science data for each glider profile

    The measurements of each glider profile are grouped into depth bins of bin_size meters
    from the surface down, and the mean, standard deviation, and number of measurements are
    calculated in each depth bin. The bins are [0, bin_size), [bin_size, 2 * bin_size), ...,
    labeled by their center depth, down to the bin containing the deepest measurement (or
    max_depth). The velocities and science variables are binned onto the same depth bins.

    ADCP velocities (velocity_variables, on the time and bin dimensions): every velocity
    measured during the profile, from every ensemble and every ADCP cell, is binned by the
    depth of the ADCP cell it was measured in (bin_depths). Velocities are excluded where
    they are missing or where the QC flag (qc_variable) is one of bad_flags. With the
    default, the measurements that fail the TRDI QC tests (4) or are missing (9) are
    excluded, and those that pass (1) or are suspect (3) are kept. With remove_bottom_hits,
    every cell of an ensemble from its first bad cell onward is also excluded, where a bad
    cell is one with a QC flag in bad_flags or a missing value for any of the velocity
    variables. The ADCP looks down, so the cells farther from the glider are deeper, and the
    good cells beyond a bad cell are likely to be reflections off the bottom rather than
    water velocities.

    Science variables (science_variables, on the time dimension): every measurement during
    the profile is binned by the depth of the glider (the depth variable), which is the
    depth of the CTD and the other science sensors. Only missing values are excluded; the
    science data are not QC'd here.

    The standard deviation is the sample standard deviation (with N - 1 degrees of
    freedom) of the measurements in the bin, so it is NaN for a bin with one measurement.
    Bins without any measurements have a mean and standard deviation of NaN and a count
    of 0.

    Each binned profile is positioned at the start of the glider profile: its time,
    latitude, and longitude are those of the profile's first measurement. The
    per-profile variables in PROFILE_VARIABLES that are in the dataset, such as the
    transect, along_distance, the deviation flags, and the depth-averaged current from
    the glider's displacement, are carried over as their value at the profile's first
    measurement.

    Parameters
    ----------
    ds: xarray.Dataset
        The glider DVL dataset with profile_id, lat/lon, and the depth variable on the time
        dimension, and bin_depths and the velocity variables on the time and bin dimensions
    bin_size: float, Default = 2
        The height (m) of the depth bins
    max_depth: float, Optional
        The deepest depth (m) to bin to. Default is the deepest measurement.
    velocity_variables: tuple[str], Default = VELOCITY_VARIABLES
        The ADCP velocity variables to bin. Pass () to bin only the science variables.
    science_variables: tuple[str], Optional
        The science variables to bin. Default is those in SCIENCE_VARIABLES that are in
        the dataset. Pass () to bin only the velocities.
    depth: str, Default = "depth_from_pressure"
        The depth (m) of the glider, used to bin the science variables
    qc_variable: str, Default = "vendor_qc_flag"
        The QC flag on the time and bin dimensions used to exclude velocities. Set to
        None to only exclude missing velocities.
    bad_flags: tuple[int], Default = (4, 9)
        The QC flag values of the velocities to exclude
    remove_bottom_hits: bool, Default = True
        Exclude every cell of an ensemble from its first bad cell onward
    lat, lon: str
        The names of the latitude and longitude variables

    Returns
    -------
    profiles: xarray.Dataset
        The profiles on the profile and depth dimensions, with:
            - profile_id, time, lat, lon (profile): the profile and its start
            - depth (depth): the center depth of each bin, with the bin edges in
              depth_bounds (depth, nv)
            - {variable}_mean, {variable}_std, {variable}_count (profile, depth): the
              mean, standard deviation, and number of measurements in each bin, for each
              velocity and science variable
            - the per-profile variables (profile) from PROFILE_VARIABLES
    """
    if science_variables is None:
        science_variables = tuple(name for name in SCIENCE_VARIABLES if name in ds)

    # Order the measurements by profile and time
    order = np.lexsort((ds["time"].values, ds["profile_id"].values))
    profile_ids, profile_idx = np.unique(ds["profile_id"].values[order], return_inverse=True)
    n_profiles = profile_ids.size

    # Depths of the ADCP cells and of the glider, and the depth bins covering both
    cell_depths = ds["bin_depths"].transpose("time", "bin").values[order].astype(float)
    glider_depth = ds[depth].values[order].astype(float)
    if max_depth is None:
        deepest = []
        if velocity_variables:
            deepest.append(np.nanmax(cell_depths))
        if science_variables:
            deepest.append(np.nanmax(glider_depth))
        max_depth = max(deepest)
    n_bins = int(np.floor(max_depth / bin_size)) + 1
    edges = np.arange(n_bins + 1) * bin_size

    profiles = xr.Dataset(
        coords={
            "profile_id": ("profile", profile_ids),
            "depth": (
                "depth",
                edges[:-1] + bin_size / 2,
                {
                    "long_name": "Depth",
                    "units": "m",
                    "positive": "down",
                    "comment": "Center depth of the depth bin",
                    "bounds": "depth_bounds",
                },
            ),
        }
    )
    profiles["depth_bounds"] = (("depth", "nv"), np.column_stack([edges[:-1], edges[1:]]))
    profiles["depth_bounds"].attrs = {"long_name": "Depth Bin Edges", "units": "m"}

    # ADCP velocities, binned by the depth of each cell
    if velocity_variables:
        cell, in_range = _bin_index(cell_depths, profile_idx[:, None], bin_size, n_bins)
        good = (
            in_range
            & good_velocity_cells(
                ds, velocity_variables, qc_variable, bad_flags, remove_bottom_hits
            )[order]
        )

        for name in velocity_variables:
            values = ds[name].transpose("time", "bin").values[order].astype(float)
            stats = _bin_statistics(cell, values, good, n_profiles, n_bins)
            _add_statistics(profiles, name, ds[name].attrs, *stats)

    # Science variables, binned by the depth of the glider
    if science_variables:
        cell, in_range = _bin_index(glider_depth, profile_idx, bin_size, n_bins)
        for name in science_variables:
            values = ds[name].values[order].astype(float)
            stats = _bin_statistics(cell, values, in_range, n_profiles, n_bins)
            _add_statistics(profiles, name, ds[name].attrs, *stats)

    profiles = add_profile_start(profiles, ds, lat, lon)
    profiles.attrs = {
        "comment": (
            f"Glider ADCP velocity and science data binned by depth into {bin_size} m bins "
            "for each glider profile"
        ),
        "bin_size": bin_size,
        "velocity_variables": " ".join(velocity_variables),
        "science_variables": " ".join(science_variables),
        "excluded_qc_flags": str(list(bad_flags)) if qc_variable is not None else "none",
        "bottom_hits_removed": str(remove_bottom_hits),
    }
    return profiles


def add_profile_start(
    profiles: xr.Dataset, ds: xr.Dataset, lat: str = "lat", lon: str = "lon"
) -> xr.Dataset:
    """Position each profile at its start, with the per-profile variables

    Adds the time, latitude, and longitude of each profile's first measurement, and
    the per-profile variables in PROFILE_VARIABLES that are in the dataset (such as
    the transect, along_distance, the deviation flags, and the depth-averaged current)
    as their value at the profile's first measurement.

    Parameters
    ----------
    profiles: xarray.Dataset
        A dataset with profile_id on the profile dimension
    ds: xarray.Dataset
        The glider DVL dataset the profiles were made from, with profile_id on the time
        dimension
    lat, lon: str
        The names of the latitude and longitude variables

    Returns
    -------
    profiles: xarray.Dataset
        The profiles with time, lat, lon, and the per-profile variables on the profile
        dimension
    """
    order = np.lexsort((ds["time"].values, ds["profile_id"].values))
    ids = ds["profile_id"].values[order]
    first = order[np.r_[True, ids[1:] != ids[:-1]]]
    position = np.searchsorted(ds["profile_id"].values[first], profiles["profile_id"].values)
    start = first[position]

    profiles["time"] = (
        "profile",
        ds["time"].values[start],
        {"long_name": "Time", "comment": "Time of the profile's first measurement"},
    )
    for name, source in [("lat", lat), ("lon", lon)]:
        profiles[name] = (
            "profile",
            ds[source].values[start],
            {**ds[source].attrs, "comment": "Position at the profile's first measurement"},
        )
    for name in PROFILE_VARIABLES:
        if name in ds and ds[name].dims == ("time",):
            profiles[name] = ("profile", ds[name].values[start], ds[name].attrs)
    return profiles


def good_velocity_cells(
    ds: xr.Dataset,
    velocity_variables: tuple[str, ...] = VELOCITY_VARIABLES,
    qc_variable: str | None = "vendor_qc_flag",
    bad_flags: tuple[int, ...] = (4, 9),
    remove_bottom_hits: bool = True,
) -> np.ndarray:
    """Find the ADCP cells with good velocities

    A cell is bad if its QC flag (qc_variable) is one of bad_flags or any of the
    velocity variables is missing. With remove_bottom_hits, every cell of an ensemble
    from its first bad cell onward is also bad: the ADCP looks down, so the cells
    farther from the glider are deeper, and the good cells beyond a bad cell are likely
    to be reflections off the bottom rather than water velocities.

    Parameters
    ----------
    ds: xarray.Dataset
        The glider DVL dataset with the velocity and QC variables on the time and bin
        dimensions
    velocity_variables: tuple[str], Default = VELOCITY_VARIABLES
        The velocity variables that must all be present for a cell to be good
    qc_variable: str, Default = "vendor_qc_flag"
        The QC flag variable. Set to None to only exclude missing velocities.
    bad_flags: tuple[int], Default = (4, 9)
        The QC flag values of the bad cells
    remove_bottom_hits: bool, Default = True
        Exclude every cell of an ensemble from its first bad cell onward

    Returns
    -------
    good: numpy.ndarray[bool]
        Whether each cell is good, on the (time, bin) dimensions
    """
    good = np.ones((ds.sizes["time"], ds.sizes["bin"]), dtype=bool)
    if qc_variable is not None:
        good &= ~np.isin(ds[qc_variable].transpose("time", "bin").values, bad_flags)
    for name in velocity_variables:
        good &= np.isfinite(ds[name].transpose("time", "bin").values)
    if remove_bottom_hits:
        good &= np.cumsum(~good, axis=1) == 0
    return good


def _bin_index(depths, profile_idx, bin_size, n_bins):
    """Index of each measurement's (profile, depth bin), and whether it is in a depth bin"""
    depth_idx = np.floor(depths / bin_size)
    in_range = np.isfinite(depth_idx) & (depth_idx >= 0) & (depth_idx < n_bins)
    cell = np.broadcast_to(profile_idx, depths.shape) * n_bins
    cell = cell + np.where(in_range, depth_idx, 0).astype(int)
    return cell, in_range


def _bin_statistics(cell, values, keep, n_profiles, n_bins):
    """Mean, sample standard deviation, and count of the kept, finite values in each bin"""
    valid = keep & np.isfinite(values)
    size = n_profiles * n_bins
    count = np.bincount(cell[valid], minlength=size)
    total = np.bincount(cell[valid], weights=values[valid], minlength=size)
    squares = np.bincount(cell[valid], weights=values[valid] ** 2, minlength=size)

    with np.errstate(invalid="ignore", divide="ignore"):
        mean = total / count
        variance = (squares - count * mean**2) / (count - 1)
    std = np.sqrt(np.clip(variance, 0, None))
    std[count < 2] = np.nan

    shape = (n_profiles, n_bins)
    return mean.reshape(shape), std.reshape(shape), count.reshape(shape)


def _add_statistics(profiles, name, attrs, mean, std, count):
    """Add the binned mean, standard deviation, and count of a variable to the profiles"""
    long_name = attrs.get("long_name", name)
    units = attrs.get("units", "")
    dims = ("profile", "depth")
    profiles[f"{name}_mean"] = (
        dims,
        mean,
        {
            "long_name": f"Mean {long_name}",
            "units": units,
            "comment": f"Mean of the {name} measurements in the depth bin",
        },
    )
    profiles[f"{name}_std"] = (
        dims,
        std,
        {
            "long_name": f"Standard Deviation of {long_name}",
            "units": units,
            "comment": f"Sample standard deviation of the {name} measurements in the depth bin",
        },
    )
    profiles[f"{name}_count"] = (
        dims,
        count,
        {
            "long_name": f"Number of {long_name} Measurements",
            "comment": f"Number of {name} measurements in the depth bin",
        },
    )
