"""
@author Andrew Reed
@brief Provides shared methods for working with glider datasets
"""

import numpy as np


def unix_time(dt, unit="s"):
    """Convert numpy datetime64 object to unix time"""
    return (dt - np.datetime64("1970-01-01T00:00:00")) / np.timedelta64(1, unit)


def magnetic_declination(
    ds,
    raw=("water_velocity_east", "water_velocity_north"),
    corrected=None,
) -> float:
    """Estimate the magnetic declination OOI applied to the ADCP water velocities

    OOI assumes the raw earth coordinate velocities the glider ADCP reports
    (water_velocity_east/north) are relative to magnetic north, and rotates them by the
    magnetic declination to relative to true north (eastward/northward_seawater_velocity).
    The declination is the median angle from the raw to the corrected velocities over
    every valid measurement, which is the rotation OOI applied. It is positive east of
    north (e.g. about -14 degrees at the Pioneer NES array, where magnetic north is west
    of true north).

    The glider ADCP has no compass of its own: it uses the glider's heading (m_heading),
    which the glider has already corrected to relative to true north (its dead reckoning
    uses m_heading directly in its true north local mission coordinates). The raw ADCP
    velocities are then already relative to true north, and the OOI rotation is a double
    correction; merge.true_north_velocities uses the raw velocities instead. For GL379
    deployment 8, the depth-averaged currents from the GPS confirm this: the profiles
    referenced to bottom tracking match them in direction without the rotation (to about
    2 degrees), and are rotated by about 14 degrees with it.

    Parameters
    ----------
    ds: xarray.Dataset
        A dataset with both the raw and corrected ADCP water velocities
    raw, corrected: tuple[str, str]
        The names of the raw and corrected eastward and northward velocities. The
        default corrected velocities are OOINet's, ooi_eastward/northward_seawater_velocity
        if merge.true_north_velocities has kept them, otherwise
        eastward/northward_seawater_velocity.

    Returns
    -------
    declination: float
        The magnetic declination (degrees)
    """
    if corrected is None:
        prefix = "ooi_" if "ooi_eastward_seawater_velocity" in ds else ""
        corrected = (f"{prefix}eastward_seawater_velocity", f"{prefix}northward_seawater_velocity")
    raw_angle = np.arctan2(ds[raw[0]].values, ds[raw[1]].values)
    corrected_angle = np.arctan2(ds[corrected[0]].values, ds[corrected[1]].values)
    rotation = np.angle(np.exp(1j * (corrected_angle - raw_angle)))
    speed = np.hypot(ds[raw[0]].values, ds[raw[1]].values)
    # Exclude near-zero velocities, whose direction is poorly defined
    valid = np.isfinite(rotation) & (speed > 0)
    return float(np.rad2deg(np.median(rotation[valid])))


def rotate_vector(east, north, angle):
    """Rotate eastward and northward components clockwise by an angle (degrees)

    Rotating a vector relative to magnetic north by the magnetic declination gives the
    vector relative to true north, and rotating by minus the declination undoes it.
    """
    theta = np.deg2rad(angle)
    cos, sin = np.cos(theta), np.sin(theta)
    return east * cos + north * sin, north * cos - east * sin
