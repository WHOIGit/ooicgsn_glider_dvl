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
    corrected=("eastward_seawater_velocity", "northward_seawater_velocity"),
) -> float:
    """Estimate the magnetic declination OOI applied to the ADCP water velocities

    The glider ADCP measures its heading with a magnetic compass, so the raw earth
    coordinate velocities it reports (water_velocity_east/north) are relative to
    magnetic north. OOI rotates them by the magnetic declination to relative to true
    north (eastward/northward_seawater_velocity). The bottom-track velocities and
    the glider's heading (m_heading) are left relative to magnetic north, so they need
    the same rotation to be combined with the water velocities.

    The declination is the median angle from the raw to the corrected velocities over
    every valid measurement, which is the rotation OOI applied. It is positive east
    of north, so a true direction is the magnetic direction plus the declination
    (e.g. about -14 degrees at the Pioneer NES array, where magnetic north is west of
    true north).

    Parameters
    ----------
    ds: xarray.Dataset
        A dataset with both the raw and corrected ADCP water velocities
    raw, corrected: tuple[str, str]
        The names of the raw and corrected eastward and northward velocities

    Returns
    -------
    declination: float
        The magnetic declination (degrees)
    """
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
    vector relative to true north.
    """
    theta = np.deg2rad(angle)
    cos, sin = np.cos(theta), np.sin(theta)
    return east * cos + north * sin, north * cos - east * sin
