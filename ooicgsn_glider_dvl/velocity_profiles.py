"""
@author Andrew Reed
@brief Provides methods for constructing water velocity profiles from the glider ADCP data by
    binning the velocities of each glider profile by depth
"""

import numpy as np
import xarray as xr

# Velocities binned by default
VELOCITY_VARIABLES = ("eastward_seawater_velocity", "northward_seawater_velocity")

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


def bin_velocity_profiles(
    ds: xr.Dataset,
    bin_size: float = 2,
    max_depth: float | None = None,
    variables: tuple[str, ...] = VELOCITY_VARIABLES,
    qc_variable: str | None = "vendor_qc_flag",
    bad_flags: tuple[int, ...] = (4, 9),
    remove_bottom_hits: bool = True,
    lat: str = "lat",
    lon: str = "lon",
) -> xr.Dataset:
    """Construct a velocity profile for each glider profile by binning the ADCP data by depth

    Every velocity measured during a glider profile, from every ensemble and every ADCP
    cell, is grouped into depth bins of bin_size meters from the surface down, using the
    depth of the ADCP cell it was measured in (bin_depths). The mean, standard deviation,
    and number of measurements are calculated in each depth bin. The bins are
    [0, bin_size), [bin_size, 2 * bin_size), ..., labeled by their center depth, down to
    the bin containing the deepest measurement (or max_depth).

    Measurements are excluded where the velocity is missing or where the QC flag
    (qc_variable) is one of bad_flags. With the default, the measurements that fail the
    TRDI QC tests (4) or are missing (9) are excluded, and those that pass (1) or are
    suspect (3) are kept.

    With remove_bottom_hits, every cell of an ensemble from its first bad cell onward is
    also excluded, where a bad cell is one with a QC flag in bad_flags or a missing value
    for any of the velocity variables. The ADCP looks down, so the cells farther from the
    glider are deeper, and the good cells beyond a bad cell are likely to be reflections
    off the bottom rather than water velocities.

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
        The glider DVL dataset with profile_id and lat/lon on the time dimension, and
        bin_depths and the velocity variables on the time and bin dimensions
    bin_size: float, Default = 2
        The height (m) of the depth bins
    max_depth: float, Optional
        The deepest depth (m) to bin to. Default is the deepest measurement.
    variables: tuple[str], Default = VELOCITY_VARIABLES
        The velocity variables to bin
    qc_variable: str, Default = "vendor_qc_flag"
        The QC flag on the time and bin dimensions used to exclude measurements. Set to
        None to only exclude missing velocities.
    bad_flags: tuple[int], Default = (4, 9)
        The QC flag values of the measurements to exclude
    remove_bottom_hits: bool, Default = True
        Exclude every cell of an ensemble from its first bad cell onward
    lat, lon: str
        The names of the latitude and longitude variables

    Returns
    -------
    profiles: xarray.Dataset
        The velocity profiles on the profile and depth dimensions, with:
            - profile_id, time, lat, lon (profile): the profile and its start
            - depth (depth): the center depth of each bin, with the bin edges in
              depth_bounds (depth, nv)
            - {variable}_mean, {variable}_std, {variable}_count (profile, depth): the
              mean, standard deviation, and number of measurements in each bin
            - the per-profile variables (profile) from PROFILE_VARIABLES
    """
    # Order the ensembles by profile, so each profile's first measurement comes first
    order = np.lexsort((ds["time"].values, ds["profile_id"].values))
    profile_id = ds["profile_id"].values[order]
    profile_ids, profile_idx = np.unique(profile_id, return_inverse=True)
    first = np.flatnonzero(np.r_[True, profile_idx[1:] != profile_idx[:-1]])
    n_profiles = profile_ids.size

    # Depth bin of every measurement
    bin_depths = ds["bin_depths"].transpose("time", "bin").values[order].astype(float)
    if max_depth is None:
        max_depth = np.nanmax(bin_depths)
    n_bins = int(np.floor(max_depth / bin_size)) + 1
    edges = np.arange(n_bins + 1) * bin_size
    depth_idx = np.floor(bin_depths / bin_size)
    in_range = np.isfinite(depth_idx) & (depth_idx >= 0) & (depth_idx < n_bins)

    # Index of each measurement's (profile, depth bin), and the measurements to keep
    cell = np.broadcast_to(profile_idx[:, None], bin_depths.shape) * n_bins
    cell = cell + np.where(in_range, depth_idx, 0).astype(int)
    good = np.ones(bin_depths.shape, dtype=bool)
    if qc_variable is not None:
        flags = ds[qc_variable].transpose("time", "bin").values[order]
        good &= ~np.isin(flags, bad_flags)

    # Exclude every cell from the first bad cell (failed QC or missing velocity) onward
    if remove_bottom_hits:
        complete = good.copy()
        for name in variables:
            complete &= np.isfinite(ds[name].transpose("time", "bin").values[order])
        good &= np.cumsum(~complete, axis=1) == 0
    keep = in_range & good

    profiles = xr.Dataset(
        coords={
            "profile_id": ("profile", profile_ids),
            "depth": ("depth", edges[:-1] + bin_size / 2, {
                "long_name": "Depth",
                "units": "m",
                "positive": "down",
                "comment": "Center depth of the depth bin",
                "bounds": "depth_bounds",
            }),
        }
    )
    profiles["depth_bounds"] = (("depth", "nv"), np.column_stack([edges[:-1], edges[1:]]))
    profiles["depth_bounds"].attrs = {"long_name": "Depth Bin Edges", "units": "m"}

    # Bin each velocity: the mean and standard deviation from the sums in each bin
    for name in variables:
        values = ds[name].transpose("time", "bin").values[order].astype(float)
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
        attrs = ds[name].attrs
        long_name = attrs.get("long_name", name)
        profiles[f"{name}_mean"] = (("profile", "depth"), mean.reshape(shape), {
            "long_name": f"Mean {long_name}",
            "units": attrs.get("units", ""),
            "comment": f"Mean of the {name} measurements in the depth bin",
        })
        profiles[f"{name}_std"] = (("profile", "depth"), std.reshape(shape), {
            "long_name": f"Standard Deviation of {long_name}",
            "units": attrs.get("units", ""),
            "comment": f"Sample standard deviation of the {name} measurements in the depth bin",
        })
        profiles[f"{name}_count"] = (("profile", "depth"), count.reshape(shape), {
            "long_name": f"Number of {long_name} Measurements",
            "comment": f"Number of {name} measurements in the depth bin",
        })

    # Position each profile at its start, with the per-profile variables
    profiles["time"] = ("profile", ds["time"].values[order][first], {
        "long_name": "Time", "comment": "Time of the profile's first measurement",
    })
    for name, source in [("lat", lat), ("lon", lon)]:
        profiles[name] = ("profile", ds[source].values[order][first], {
            **ds[source].attrs, "comment": "Position at the profile's first measurement",
        })
    for name in PROFILE_VARIABLES:
        if name in ds and ds[name].dims == ("time",):
            profiles[name] = ("profile", ds[name].values[order][first], ds[name].attrs)

    profiles.attrs = {
        "comment": (
            f"Velocity profiles from the glider ADCP, binned by depth into {bin_size} m bins "
            "for each glider profile"
        ),
        "bin_size": bin_size,
        "excluded_qc_flags": str(list(bad_flags)) if qc_variable is not None else "none",
        "bottom_hits_removed": str(remove_bottom_hits),
    }
    return profiles
