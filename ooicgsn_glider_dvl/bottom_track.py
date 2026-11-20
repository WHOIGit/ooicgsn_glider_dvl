"""
@author Andrew Reed
@brief Provides methods for removing the glider's motion from the DVL water velocities using
    bottom tracking, and for using the bottom-tracked velocities to check the velocities from
    the flight model
"""

import numpy as np
import pandas as pd
import xarray as xr

from ooicgsn_glider_dvl.binning import VELOCITY_VARIABLES
from ooicgsn_glider_dvl.utils import rotate_vector

# The water velocities with the bottom track removed, added by remove_bottom_track
BOTTOM_TRACK_VELOCITY_VARIABLES = (
    "bt_eastward_seawater_velocity",
    "bt_northward_seawater_velocity",
)


def add_bottom_track(
    ds: xr.Dataset,
    bottom_track: xr.Dataset,
    declination: float = 0,
    max_error_velocity: float | None = None,
) -> xr.Dataset:
    """Add the bottom-track velocities to the glider DVL dataset

    When the bottom is in range, the DVL measures the velocity of the bottom relative to
    the glider, which is the opposite of the glider's velocity over the ground. The
    bottom-track velocities are in earth coordinates, using the glider's heading, which
    is already relative to true north, and in mm s-1. They are converted to m s-1, and
    only rotated by the declination, if one is given. They are then in the same frame as
    the raw DVL water velocities (see merge.true_north_velocities).

    Parameters
    ----------
    ds: xarray.Dataset
        The glider DVL dataset
    bottom_track: xarray.Dataset
        The OOI glider ADCP bottom-track dataset, on the same times as the DVL dataset,
        with bt_eastward_velocity, bt_northward_velocity, bt_upward_velocity, and
        bt_error_velocity (mm s-1)
    declination: float, Default = 0
        A magnetic declination (degrees, positive east of north) to rotate the velocities
        by, for an ADCP whose heading is relative to magnetic north
    max_error_velocity: float, Optional
        Exclude the bottom-track velocities with an error velocity (m s-1) larger than
        this, up or down

    Returns
    -------
    ds: xarray.Dataset
        The dataset with bottom_track_eastward_velocity, bottom_track_northward_velocity,
        and bottom_track_upward_velocity (m s-1) on the time dimension, which are NaN
        where there is no bottom track
    """
    bt = bottom_track.reindex(time=ds["time"], method="nearest", tolerance=np.timedelta64(1, "s"))
    east = bt["bt_eastward_velocity"].values / 1000
    north = bt["bt_northward_velocity"].values / 1000
    up = bt["bt_upward_velocity"].values / 1000
    east, north = rotate_vector(east, north, declination)

    valid = np.isfinite(east) & np.isfinite(north) & np.isfinite(up)
    if max_error_velocity is not None:
        error = bt["bt_error_velocity"].values / 1000
        valid &= np.abs(error) <= max_error_velocity

    for name, values in [("eastward", east), ("northward", north), ("upward", up)]:
        ds[f"bottom_track_{name}_velocity"] = (
            "time",
            np.where(valid, values, np.nan),
            {
                "long_name": f"Bottom Track {name.capitalize()} Velocity",
                "units": "m s-1",
                "comment": (
                    f"{name.capitalize()} velocity of the bottom relative to the glider, the "
                    "opposite of the glider's velocity over the ground"
                    + (
                        f", rotated by a declination of {declination:.2f} degrees"
                        if declination and name != "upward"
                        else ""
                    )
                ),
                "source_sensor": f"bt_{name}_velocity",
            },
        )
    return ds


def remove_bottom_track(
    ds: xr.Dataset,
    velocity_variables: tuple[str, str] = VELOCITY_VARIABLES,
) -> xr.Dataset:
    """Remove the glider's motion from the DVL water velocities using the bottom track

    The DVL water velocities are relative to the glider. Subtracting the bottom-track
    velocity (the velocity of the bottom relative to the glider) gives the absolute
    water velocity, wherever there is bottom tracking.

    Parameters
    ----------
    ds: xarray.Dataset
        The glider DVL dataset with the bottom-track velocities (see add_bottom_track)
    velocity_variables: tuple[str, str]
        The eastward and northward water velocities relative to the glider (time, bin)

    Returns
    -------
    ds: xarray.Dataset
        The dataset with the absolute water velocities BOTTOM_TRACK_VELOCITY_VARIABLES on
        the time and bin dimensions, which are NaN where there is no bottom track
    """
    bottom_track = ["bottom_track_eastward_velocity", "bottom_track_northward_velocity"]
    for name, relative, bt in zip(
        BOTTOM_TRACK_VELOCITY_VARIABLES, velocity_variables, bottom_track
    ):
        long_name = ds[relative].attrs.get("long_name", relative)
        ds[name] = ds[relative] - ds[bt]
        ds[name].attrs = {
            **ds[relative].attrs,
            "long_name": f"Bottom-Tracked {long_name}",
            "comment": (
                f"{long_name} with the glider's motion removed by subtracting the bottom-track "
                "velocity. Missing where there is no bottom tracking."
            ),
            "ancillary_variables": f"{relative} {bt}",
        }
    return ds


def compare_glider_velocity(ds: xr.Dataset, ground: xr.Dataset) -> pd.DataFrame:
    """Compare the glider's velocity over the ground from the inversion with bottom tracking

    The inversion (inversion.invert_profiles) solves for the glider's velocity over the
    ground from the DVL, the flight model, and the depth-averaged current. Bottom
    tracking measures it directly (the opposite of the bottom-track velocity),
    independently of the flight model and the depth-averaged current, so the difference
    is the error of the inversion at the glider's depth.

    Parameters
    ----------
    ds: xarray.Dataset
        The glider DVL dataset with the bottom-track velocities (see add_bottom_track)
    ground: xarray.Dataset
        The glider velocity over the ground from inversion.invert_profiles

    Returns
    -------
    comparison: pandas.DataFrame
        One row per measurement with both, with the time, profile_id, the glider's
        eastward and northward velocity over the ground from the inversion (inversion_*)
        and bottom tracking (bt_*), and their difference (inversion minus bottom
        tracking, difference_*)
    """
    inverted = np.column_stack(
        [
            ground["glider_eastward_ground_velocity"].values,
            ground["glider_northward_ground_velocity"].values,
        ]
    )
    bt = -np.column_stack(
        [ds["bottom_track_eastward_velocity"].values, ds["bottom_track_northward_velocity"].values]
    )
    valid = np.isfinite(inverted).all(axis=1) & np.isfinite(bt).all(axis=1)
    return pd.DataFrame(
        {
            "time": ds["time"].values[valid],
            "profile_id": ds["profile_id"].values[valid],
            "inversion_eastward": inverted[valid, 0],
            "inversion_northward": inverted[valid, 1],
            "bt_eastward": bt[valid, 0],
            "bt_northward": bt[valid, 1],
            "difference_eastward": inverted[valid, 0] - bt[valid, 0],
            "difference_northward": inverted[valid, 1] - bt[valid, 1],
        }
    )


def split_by_depth(
    ds: xr.Dataset,
    use: np.ndarray,
    span: float = 20,
    depth: str = "depth_from_pressure",
    profile: str = "profile_id",
) -> tuple[np.ndarray, np.ndarray]:
    """Split each dive's bottom tracking into a deep reference part and the part above it

    The DVL only tracks the bottom near the end of each dive (within about 60 m of the
    bottom), so profiles referenced to bottom tracking carry the reference up the water
    column with the DVL shear and the flight model. To test how well, each dive's bottom
    track is split by the glider's depth: the reference is the bottom tracking within span
    of the deepest glider depth with bottom tracking, and the rest is held out, at a
    distance above the shallowest reference depth.

    Parameters
    ----------
    ds: xarray.Dataset
        The glider DVL dataset with the profile and depth variables
    use: numpy.ndarray[bool]
        The ensembles with bottom tracking to split, e.g. those in steady flight
    span: float, Default = 20
        The depth range (m) of the reference at the bottom of each dive's bottom tracking
    depth, profile: str
        The glider depth (m) and profile variables

    Returns
    -------
    reference: numpy.ndarray[bool]
        The ensembles of the reference
    distance: numpy.ndarray[float]
        The height (m) of each ensemble of use above its dive's reference (the shallowest
        reference depth minus its depth), NaN where use is False
    """
    z = ds[depth].values.astype(float)
    frame = pd.DataFrame({"profile": ds[profile].values, "z": np.where(use, z, np.nan)})
    deepest = frame.groupby("profile")["z"].transform("max").to_numpy()
    reference = use & (z >= deepest - span)
    frame["reference_z"] = np.where(reference, z, np.nan)
    top = frame.groupby("profile")["reference_z"].transform("min").to_numpy()
    distance = np.where(use, top - z, np.nan)
    return reference, distance


def through_water_velocity(ds: xr.Dataset, ground: xr.Dataset) -> pd.DataFrame:
    """The glider's velocity through the water from bottom tracking and the velocity profile

    The glider's velocity through the water is its velocity over the ground minus the
    absolute water velocity at its depth. With the profiles solved from the DVL and
    bottom tracking alone (inversion.invert_profiles with dac_weight=0 and
    flight_weight=0), the velocity over the ground of the ensembles with bottom tracking is
    the bottom-track velocity, and the water velocity at the glider's depth comes from the
    cells of earlier ensembles, which measured the water below the glider before it
    descended into it (Merckelbach et al., 2019). This is independent of the flight model
    and of the DVL cell nearest the glider, so it can check both.

    Parameters
    ----------
    ds: xarray.Dataset
        The glider DVL dataset with vehicle_heading and the bottom-track velocities (see
        add_bottom_track)
    ground: xarray.Dataset
        The ground velocity output of invert_profiles, on the same times as ds

    Returns
    -------
    velocity: pandas.DataFrame
        For each ensemble with bottom tracking and a solved profile, the time, the
        eastward and northward velocity through the water, and its components along the
        glider's heading (forward) and to its starboard (sideways), in m s-1
    """
    east = (
        ground["glider_eastward_ground_velocity"].values
        - ground["eastward_velocity_at_glider"].values
    )
    north = (
        ground["glider_northward_ground_velocity"].values
        - ground["northward_velocity_at_glider"].values
    )
    heading = np.deg2rad(ds["vehicle_heading"].values)
    has_bottom_track = np.isfinite(ds["bottom_track_eastward_velocity"].values)
    valid = has_bottom_track & np.isfinite(east) & np.isfinite(north)
    return pd.DataFrame(
        {
            "time": ds["time"].values[valid],
            "eastward": east[valid],
            "northward": north[valid],
            "forward": (east * np.sin(heading) + north * np.cos(heading))[valid],
            "sideways": (east * np.cos(heading) - north * np.sin(heading))[valid],
        }
    )


def compare_profiles(
    model: xr.Dataset,
    bottom_tracked: xr.Dataset,
    variables: tuple[tuple[str, str], ...] = (
        ("eastward_seawater_velocity", "bt_eastward_seawater_velocity_mean"),
        ("northward_seawater_velocity", "bt_northward_seawater_velocity_mean"),
    ),
    count: str = "bt_eastward_seawater_velocity_count",
    min_count: int = 3,
) -> xr.Dataset:
    """Compare the velocity profiles from the inversion with the bottom-tracked profiles

    The profiles from the DVL, flight model, and depth-averaged current
    (inversion.invert_profiles) are compared with the bottom-tracked profiles
    (binning.bin_profiles of the BOTTOM_TRACK_VELOCITY_VARIABLES) on their common
    profiles and depth bins, where the bottom-tracked bin has at least min_count
    measurements. The difference of each profile is split into its depth-mean (offset),
    the error of the reference to the depth-averaged current, and the remainder, the
    error of the shape of the profile.

    Parameters
    ----------
    model: xarray.Dataset
        The profiles from inversion.invert_profiles
    bottom_tracked: xarray.Dataset
        The bottom-tracked profiles from binning.bin_profiles
    variables: tuple[tuple[str, str]]
        The pairs of model and bottom-tracked variables to compare
    count: str
        The variable with the number of bottom-tracked measurements in each bin
    min_count: int, Default = 3
        The fewest bottom-tracked measurements in a depth bin to compare it

    Returns
    -------
    comparison: xarray.Dataset
        On the profile_id and depth dimensions of the common profiles and depth bins, for
        each pair of variables, named by the model variable:
            - {variable}_difference (profile_id, depth): model minus bottom-tracked
            - {variable}_offset (profile_id): the mean difference over the depth bins
            - {variable}_shape_rms (profile_id): the rms difference after removing the
              offset
            - {variable}_bias, {variable}_rms (depth): the mean and rms difference over
              the profiles
        with the number of compared depth bins of each profile (n_bins) and the number of
        compared profiles at each depth (n_profiles)
    """
    # Match the profiles by profile_id and the depth bins by depth
    model = model.swap_dims(profile="profile_id")
    bottom_tracked = bottom_tracked.swap_dims(profile="profile_id")
    model, bottom_tracked = xr.align(model, bottom_tracked, join="inner")
    keep = bottom_tracked[count] >= min_count

    comparison = xr.Dataset(coords={"profile_id": model["profile_id"], "depth": model["depth"]})
    for model_name, bt_name in variables:
        difference = (model[model_name] - bottom_tracked[bt_name]).where(keep)
        offset = difference.mean("depth")
        units = {"units": "m s-1"}
        comparison[f"{model_name}_difference"] = difference.assign_attrs(units)
        comparison[f"{model_name}_offset"] = offset.assign_attrs(units)
        comparison[f"{model_name}_shape_rms"] = np.sqrt(
            ((difference - offset) ** 2).mean("depth")
        ).assign_attrs(units)
        comparison[f"{model_name}_bias"] = difference.mean("profile_id").assign_attrs(units)
        comparison[f"{model_name}_rms"] = np.sqrt((difference**2).mean("profile_id")).assign_attrs(
            units
        )
        valid = difference.notnull()
    comparison["n_bins"] = valid.sum("depth")
    comparison["n_profiles"] = valid.sum("profile_id")
    return comparison
