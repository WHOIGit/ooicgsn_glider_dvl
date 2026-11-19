"""
@author Andrew Reed
@brief Provides corrections to the glider DVL data: the offset between the DVL and glider clocks,
    and the sound speed used by the DVL to calculate the vertical velocity
"""

import gsw
import numpy as np
import pandas as pd
import xarray as xr

# Variables in the OOINet ADCP datasets that OOINet matched to the DVL times from the glider data,
# rather than measured by the DVL, which are recalculated at the corrected times by shift_time
PRESSURE_VARIABLES = ("int_ctd_pressure", "non_zero_pressure")
DEPTH_VARIABLES = (
    "depth_from_pressure",
    "non_zero_depth",
    "ctdgv_m_glider_instrument_recovered-depth",
)
POSITION_VARIABLES = ("lat", "lon")


def _seconds(time) -> np.ndarray:
    return np.asarray(time).astype("datetime64[ns]").astype("int64") / 1e9


def _pressure_series(
    gdac: xr.Dataset, pressure: str = "pressure"
) -> tuple[np.ndarray, np.ndarray]:
    """The times (s) and CTD pressures (dbar) of the GDAC science dataset, without missing values"""
    p = gdac[pressure].to_series().dropna()
    p = p[~p.index.duplicated()].sort_index()
    return _seconds(p.index.values), p.values.astype(float)


def bottom_altitude(bottom_track: xr.Dataset) -> np.ndarray:
    """The mean range to the bottom of the four DVL beams (m), where all four detect it"""
    ranges = np.stack([bottom_track[f"bt_beam{n}_range"].values for n in range(1, 5)]) / 100
    ranges[~(ranges > 0)] = np.nan
    altitude = np.full(ranges.shape[1], np.nan)
    valid = np.isfinite(ranges).all(axis=0)
    altitude[valid] = ranges[:, valid].mean(axis=0)
    return altitude


def estimate_time_offset(
    bottom_track: xr.Dataset,
    gdac: xr.Dataset,
    profile_id: np.ndarray,
    max_lag: float = 15,
    step: float = 0.25,
    min_points: int = 20,
) -> pd.DataFrame:
    """Estimate the offset between the DVL clock and the glider science (CTD) clock

    The time stamps of the DVL ensembles are offset from those of the glider science data
    (e.g. about 3 s in Merckelbach et al., 2019). OOINet matches the glider CTD pressure,
    and so the depth, to the DVL times, so an offset puts every DVL ensemble at the wrong
    depth. The offset is estimated with a quantity the DVL measures itself, independently of
    the glider's attitude and clocks: the range to the bottom. While the bottom is in range,
    the glider's altitude above the bottom and its depth add up to the water depth, which
    changes only slowly as the glider flies over a sloping bottom. For each dive with bottom
    tracking, the altitude (the mean range of the four beams) is regressed on the CTD
    pressure at the DVL times shifted by a trial offset, and on time, for a linear change of
    the water depth along the glider's path. The offset of the dive is the one with the
    smallest residual, and the offset of the deployment is the one with the smallest sum of
    the residuals of all of the dives, each normalized by its smallest residual.

    The glider's attitude can't be used for this: the DVL records the pitch, roll, and
    heading it receives from the glider, which are delayed by the transfer from the glider
    to the DVL.

    Parameters
    ----------
    bottom_track: xarray.Dataset
        The OOINet DVL bottom track dataset, with bt_beam1_range to bt_beam4_range (cm)
    gdac: xarray.Dataset
        The IOOS GDAC glider science dataset, with the CTD pressure (dbar)
    profile_id: numpy.ndarray
        The profile of each DVL time (see profiles.get_profile_ids)
    max_lag: float, Default = 15
        The largest offset (s) to try, earlier or later
    step: float, Default = 0.25
        The step (s) between the offsets tried
    min_points: int, Default = 20
        The fewest bottom track measurements in a dive to estimate its offset

    Returns
    -------
    offsets: pandas.DataFrame
        One row per dive with the profile_id, the number of bottom track measurements, the
        best offset (s) and the rms residual (m) at it. The offset of the deployment is in
        offsets.attrs["time_offset"]. A positive offset means the DVL times are early: the
        DVL measured each ensemble that many seconds after its time stamp.
    """
    lags = np.arange(-max_lag, max_lag + step / 2, step)
    t_ctd, p_ctd = _pressure_series(gdac)
    t_dvl = _seconds(bottom_track["time"].values)
    altitude = bottom_altitude(bottom_track)
    valid = np.isfinite(altitude)

    rows, total = [], np.zeros(lags.size)
    for pid, idx in pd.Series(np.flatnonzero(valid)).groupby(np.asarray(profile_id)[valid]):
        i = idx.values
        if i.size < min_points:
            continue
        t = t_dvl[i] - t_dvl[i].mean()
        rms = np.empty(lags.size)
        for n, lag in enumerate(lags):
            design = np.column_stack([np.ones_like(t), t, np.interp(t_dvl[i] + lag, t_ctd, p_ctd)])
            residual = altitude[i] - design @ np.linalg.lstsq(design, altitude[i], rcond=None)[0]
            rms[n] = np.sqrt(np.mean(residual**2))
        best = int(np.argmin(rms))
        total += rms / rms[best]
        rows.append((pid, i.size, lags[best], rms[best]))

    offsets = pd.DataFrame(rows, columns=["profile_id", "n_points", "time_offset", "rms"])
    offsets.attrs["time_offset"] = float(lags[int(np.argmin(total))])
    return offsets


def shift_time(
    ds: xr.Dataset,
    offset: float,
    gdac: xr.Dataset | None = None,
    max_gap: float | None = 60,
) -> xr.Dataset:
    """Correct the DVL times for the offset from the glider science (CTD) clock

    The offset (see estimate_time_offset) is added to the times. The variables that OOINet
    matched to the DVL times from the glider data, rather than measured by the DVL, are then
    recalculated at the corrected times: the CTD pressure (int_ctd_pressure,
    non_zero_pressure) is interpolated from the GDAC science dataset (or from OOINet's CTD
    pressure where the GDAC dataset has a gap longer than max_gap, e.g. at the surface before
    the CTD's first measurement of a dive), the depths
    (depth_from_pressure, non_zero_depth) are calculated from it with gsw, as OOINet does,
    and the depths of the cells (bin_depths) are moved by the same change in depth. The
    interpolated position (lat, lon) is interpolated to the corrected times. Everything the
    DVL measured or recorded itself, including the attitude it received from the glider, is
    unchanged. It is skipped if the dataset has already been shifted (it has the
    dvl_time_offset attribute), so it is safe to apply more than once.

    Parameters
    ----------
    ds: xarray.Dataset
        The OOINet glider DVL dataset (ADCP or bottom track)
    offset: float
        The offset (s) to add to the DVL times
    gdac: xarray.Dataset, optional
        The IOOS GDAC glider science dataset with the CTD pressure, to recalculate the
        pressure and depth at the corrected times. Without it, they are left unchanged.
    max_gap: float or None, Default = 60
        The longest gap between CTD pressures, in seconds, to interpolate across

    Returns
    -------
    ds: xarray.Dataset
        The dataset at the corrected times
    """
    if "dvl_time_offset" in ds.attrs:
        return ds
    old = _seconds(ds["time"].values)
    new = old + offset
    ds = ds.assign_coords(time=ds["time"].values + np.timedelta64(round(offset * 1e9), "ns"))
    ds["time"].attrs = {
        "long_name": "Time",
        "comment": (
            f"The DVL ensemble time corrected for the offset of {offset:g} s from the glider "
            "science (CTD) clock; the uncorrected time is internal_timestamp"
        ),
    }

    if gdac is not None:
        from ooicgsn_glider_dvl.merge import interp_1d

        t_ctd, p_ctd = _pressure_series(gdac)
        pressure = interp_1d(new, t_ctd, p_ctd, max_gap)
        # Where the CTD is off (e.g. at the surface, before the first CTD measurement of a dive),
        # fall back to OOINet's CTD pressure, interpolated to the corrected time
        if "int_ctd_pressure" in ds:
            missing = np.isnan(pressure)
            ooi = ds["int_ctd_pressure"].values
            valid = np.isfinite(ooi)
            pressure[missing] = np.interp(new[missing], old[valid], ooi[valid])
        lat = ds["lat"].values if "lat" in ds else np.full(new.shape, 40.0)
        depth = -gsw.z_from_p(pressure, np.interp(new, old, lat))
        change = None
        for name in PRESSURE_VARIABLES:
            if name in ds:
                ds[name] = (ds[name].dims, pressure, ds[name].attrs)
        for name in DEPTH_VARIABLES:
            if name in ds:
                if name == "depth_from_pressure":
                    change = depth - ds[name].values
                ds[name] = (ds[name].dims, depth, ds[name].attrs)
        if "bin_depths" in ds and change is not None:
            ds["bin_depths"] = ds["bin_depths"] + xr.DataArray(change, dims="time")

    for name in POSITION_VARIABLES:
        if name in ds:
            ds[name] = (ds[name].dims, np.interp(new, old, ds[name].values), ds[name].attrs)

    ds.attrs["dvl_time_offset"] = offset
    return ds


def sound_speed_factor(
    ds: xr.Dataset,
    beam_angle: float = 30,
    temperature: str = "ctd_temperature",
    salinity: str = "ctd_salinity",
    pressure: str = "ctd_pressure",
) -> xr.Dataset:
    """Calculate the correction to the DVL's vertical velocities for the sound speed

    The DVL calculates the sound speed from its transducer temperature, which reads warmer
    than the water (it is in the glider's hull), and a fixed salinity (35). The sound speed
    from the glider CTD is the true sound speed. The DVL has a phased array transducer, whose
    beam angle changes with the sound speed (sin(angle) is proportional to it), so that the
    velocities along the instrument's face (horizontal, for a level instrument) don't depend
    on the sound speed, but the velocity along its axis (vertical) does. For a velocity w the
    DVL calculated along its axis with sound speed c_dvl and beam angle angle_dvl, the true
    velocity is

        w * (c / c_dvl) * cos(angle_dvl) / cos(angle),    sin(angle) = sin(angle_dvl) c / c_dvl

    assuming the DVL calculates its velocities with the beam angle for its own sound speed.
    Both sound speeds are calculated with gsw (TEOS-10) at the CTD pressure, so the pressure
    term cancels, and only the temperature and salinity differ.

    Parameters
    ----------
    ds: xarray.Dataset
        The merged glider DVL dataset, with the DVL's temperature (cdeg_C or degrees C) and
        salinity, the interpolated glider CTD temperature, salinity, and pressure, and lat
        and lon
    beam_angle: float, Default = 30
        The angle of the DVL beams from its axis (degrees) at its sound speed
    temperature, salinity, pressure: str
        The glider CTD temperature (degrees C), practical salinity, and pressure (dbar)

    Returns
    -------
    ds: xarray.Dataset
        The dataset with dvl_sound_speed, sound_speed (m s-1), and sound_speed_factor on the
        time dimension
    """
    p = ds[pressure].values
    lon, lat = ds["lon"].values, ds["lat"].values
    dvl_t = ds["temperature"].values.astype(float)
    if ds["temperature"].attrs.get("units") == "cdeg_C":
        dvl_t = dvl_t / 100

    def speed(t, sp):
        sa = gsw.SA_from_SP(sp, p, lon, lat)
        return gsw.sound_speed(sa, gsw.CT_from_t(sa, t, p), p)

    c_dvl = speed(dvl_t, ds["salinity"].values.astype(float))
    c = speed(ds[temperature].values, ds[salinity].values)
    ratio = c / c_dvl
    sin_dvl = np.sin(np.deg2rad(beam_angle))
    factor = ratio * np.sqrt(1 - sin_dvl**2) / np.sqrt(1 - (sin_dvl * ratio) ** 2)

    ds["dvl_sound_speed"] = (
        "time",
        c_dvl,
        {
            "long_name": "DVL Sound Speed",
            "units": "m s-1",
            "comment": (
                "The sound speed used by the DVL: calculated with gsw from its transducer "
                "temperature and fixed salinity, at the glider CTD pressure"
            ),
        },
    )
    ds["sound_speed"] = (
        "time",
        c,
        {
            "long_name": "Sound Speed",
            "units": "m s-1",
            "comment": "Sound speed calculated with gsw from the glider CTD",
        },
    )
    ds["sound_speed_factor"] = (
        "time",
        factor,
        {
            "long_name": "Sound Speed Correction Factor",
            "units": "1",
            "comment": (
                "Factor to correct the velocity along the DVL's axis for the sound speed, for a "
                f"phased array transducer with a beam angle of {beam_angle:g} degrees"
            ),
        },
    )
    return ds


def instrument_axis(ds: xr.Dataset, mount_angle: float = 11) -> tuple[np.ndarray, ...]:
    """The eastward, northward, and upward components of the DVL's axis (unit vector)

    The DVL looks downward, mounted with its axis pitched forward by mount_angle from the
    glider's vertical axis. It is rotated to Earth coordinates by the glider's roll, pitch
    (positive nose up), and heading (degrees true) recorded by the DVL, i.e. those it used
    to rotate its velocities to Earth coordinates.
    """
    m = np.deg2rad(mount_angle)
    roll, pitch, heading = (
        np.deg2rad(ds[v].values.astype(float)) for v in ("roll", "pitch", "heading")
    )
    # The axis in the glider's frame (forward, starboard, down), then rolled, pitched, turned
    x, y, z = np.sin(m), -np.cos(m) * np.sin(roll), np.cos(m) * np.cos(roll)
    x, z = np.cos(pitch) * x + np.sin(pitch) * z, -np.sin(pitch) * x + np.cos(pitch) * z
    north = np.cos(heading) * x - np.sin(heading) * y
    east = np.sin(heading) * x + np.cos(heading) * y
    return east, north, -z


def correct_vertical_velocity(
    ds: xr.Dataset,
    velocities: tuple[str, str, str] = (
        "eastward_seawater_velocity",
        "northward_seawater_velocity",
        "upward_seawater_velocity",
    ),
    raw: str | None = "water_velocity_up",
    mount_angle: float = 11,
) -> xr.Dataset:
    """Correct the DVL's upward velocities for the sound speed

    The component of the velocity along the DVL's axis (see instrument_axis) is scaled by the
    sound_speed_factor (see sound_speed_factor), and the upward velocity is corrected for its
    change. The components along the face of the DVL don't depend on the sound speed. As the
    DVL is tilted from the vertical, the change along its axis also has a small horizontal
    part, which is not applied: the eastward and northward velocities are unchanged.

    The upward velocity is recalculated from the raw upward velocity (raw, in mm s-1) if it
    is given, so it is safe to apply more than once. Otherwise it is skipped if the upward
    velocity is already corrected (it has the sound_speed_corrected attribute).

    Parameters
    ----------
    ds: xarray.Dataset
        The merged glider DVL dataset with the sound_speed_factor and the DVL's heading,
        pitch, and roll (degrees)
    velocities: tuple[str, str, str]
        The eastward, northward, and upward velocities (m s-1), e.g. the water velocities
        or the bottom track velocities
    raw: str or None, Default = "water_velocity_up"
        The raw upward velocity (mm s-1) to recalculate the upward velocity from
    mount_angle: float, Default = 11
        The angle (degrees) the DVL is mounted pitched forward from the glider's vertical

    Returns
    -------
    ds: xarray.Dataset
        The dataset with the corrected upward velocity
    """
    east_name, north_name, up_name = velocities
    if raw is None and ds[up_name].attrs.get("sound_speed_corrected"):
        return ds
    up = ds[raw] / 1000 if raw is not None else ds[up_name]
    a_east, a_north, a_up = (
        xr.DataArray(component, dims="time") for component in instrument_axis(ds, mount_angle)
    )
    along = ds[east_name] * a_east + ds[north_name] * a_north + up * a_up
    corrected = up + (ds["sound_speed_factor"] - 1) * along * a_up

    attrs = dict(ds[up_name].attrs)
    if not attrs.get("sound_speed_corrected"):
        attrs["comment"] = (
            f"{attrs.get('comment', '')} Corrected for the sound speed from the glider CTD: "
            "the velocity along the DVL's axis is scaled by sound_speed_factor."
        ).strip()
    attrs["sound_speed_corrected"] = 1
    attrs["mount_angle"] = mount_angle
    ds[up_name] = corrected.transpose(*ds[up_name].dims).astype(ds[up_name].dtype)
    ds[up_name].attrs = attrs
    return ds
