"""
@author Andrew Reed
@brief Provides methods for calculating the depth-averaged current from the glider's
    dead-reckoned and GPS positions at each surfacing
"""

import numpy as np
import pandas as pd
import xarray as xr

# Glider engineering variables needed to calculate the depth-averaged current
CURRENT_VARIABLES = [
    "time",
    "m_dr_time",
    "m_dr_surf_x_lmc",
    "m_dr_surf_y_lmc",
    "m_gps_fix_x_lmc",
    "m_gps_fix_y_lmc",
    "m_water_vx",
    "m_water_vy",
    "m_gps_lat",
    "m_gps_lon",
]


def surfacing_currents(eng: pd.DataFrame) -> pd.DataFrame:
    """Calculate the depth-averaged current for each glider surfacing

    Between surfacings the glider dead reckons its position from its
    heading, its speed through the water, and the current it estimated
    at the previous surfacing. When it surfaces, the displacement between
    its first GPS fix and its dead-reckoned position is the drift that the
    dead reckoning did not account for. Dividing the displacement by the
    time spent underwater gives the correction to the current estimate,
    and adding it to the current used in the dead reckoning gives the
    depth-averaged current over all of the dives since the last surfacing.

    The positions are the glider's own dead-reckoned surfacing position
    (m_dr_surf_x/y_lmc) and first GPS fix (m_gps_fix_x/y_lmc), in meters
    east and north in the glider's local mission coordinates. The m_lat
    and m_lon in the OOI engineering data are not used, because they are
    interpolated between the dead-reckoned position and the GPS fix.

    Parameters
    ----------
    eng: pandas.DataFrame
        The glider engineering data downloaded from OOI with the
        CURRENT_VARIABLES columns

    Returns
    -------
    surf: pandas.DataFrame
        One row per surfacing with the dive start and surfacing times, the
        time underwater (s), the GPS fix position, the displacement between
        the GPS fix and the dead-reckoned position (m), the current used in
        the dead reckoning, the correction to it, and the depth-averaged
        current (m/s)
    """
    eng = eng.assign(time=pd.to_datetime(eng["time"])).sort_values("time")

    # The engineering data only records a value when it changes, so the dead-reckoned position
    # (reported when the glider reaches the surface) and the GPS fix (reported a few minutes
    # later) are on separate rows. Take the first report of each.
    dr = eng.dropna(subset=["m_dr_time", "m_dr_surf_x_lmc", "m_dr_surf_y_lmc"])
    dr = dr[dr["m_dr_time"] > 0]
    dr = dr.drop_duplicates(subset=["m_dr_time", "m_dr_surf_x_lmc", "m_dr_surf_y_lmc"], keep="first")
    dr = dr[["time", "m_dr_time", "m_dr_surf_x_lmc", "m_dr_surf_y_lmc"]].rename(columns={"time": "surface_time"})

    fix = eng.dropna(subset=["m_gps_fix_x_lmc", "m_gps_fix_y_lmc"])
    fix = fix[(fix["m_gps_fix_x_lmc"] != 0) | (fix["m_gps_fix_y_lmc"] != 0)]
    fix = fix.drop_duplicates(subset=["m_gps_fix_x_lmc", "m_gps_fix_y_lmc"], keep="first")
    fix = fix[["time", "m_gps_fix_x_lmc", "m_gps_fix_y_lmc"]].rename(columns={"time": "fix_time"})

    # Pair each surfacing with the first GPS fix after it, before the next surfacing
    surf = pd.merge_asof(dr, fix, left_on="surface_time", right_on="fix_time", direction="forward")
    next_surface = surf["surface_time"].shift(-1)
    surf = surf[surf["fix_time"].notna() & ~(surf["fix_time"] >= next_surface)]
    surf["dive_start_time"] = surf["surface_time"] - pd.to_timedelta(surf["m_dr_time"], unit="s")
    surf = surf.reset_index(drop=True)

    # The current used in the dead reckoning is the glider's estimate before it surfaced
    water_v = eng[["time", "m_water_vx", "m_water_vy"]].dropna()
    used = pd.merge_asof(
        surf[["surface_time"]], water_v, left_on="surface_time", right_on="time", direction="backward"
    )

    # The latitude and longitude of the GPS fix, for the location of the current
    gps = eng[["time", "m_gps_lat", "m_gps_lon"]].dropna()
    gps = gps[(gps["m_gps_lat"] != 0) | (gps["m_gps_lon"] != 0)]
    gps_fix = pd.merge_asof(
        surf[["fix_time"]], gps, left_on="fix_time", right_on="time", direction="forward"
    )

    dt = surf["m_dr_time"].values
    out = pd.DataFrame({
        "dive_start_time": surf["dive_start_time"],
        "surface_time": surf["surface_time"],
        "time_underwater": dt,
        "gps_fix_time": surf["fix_time"],
        "gps_fix_lat": gps_fix["m_gps_lat"].values,
        "gps_fix_lon": gps_fix["m_gps_lon"].values,
        "displacement_east": (surf["m_gps_fix_x_lmc"] - surf["m_dr_surf_x_lmc"]).values,
        "displacement_north": (surf["m_gps_fix_y_lmc"] - surf["m_dr_surf_y_lmc"]).values,
        "dr_current_east": used["m_water_vx"].fillna(0).values,
        "dr_current_north": used["m_water_vy"].fillna(0).values,
    })
    out["correction_east"] = out["displacement_east"] / dt
    out["correction_north"] = out["displacement_north"] / dt
    out["current_east"] = out["dr_current_east"] + out["correction_east"]
    out["current_north"] = out["dr_current_north"] + out["correction_north"]

    return out


def add_depth_averaged_current(dvl: xr.Dataset, surf: pd.DataFrame) -> xr.Dataset:
    """Add the depth-averaged current to the glider DVL dataset

    Each DVL time between the start of a segment of dives and the
    surfacing at its end is given the depth-averaged current calculated
    at that surfacing. DVL times outside of a segment are filled with NaN.

    Parameters
    ----------
    dvl: xarray.Dataset
        The OOI glider DVL dataset
    surf: pandas.DataFrame
        The depth-averaged current at each surfacing from surfacing_currents

    Returns
    -------
    dvl: xarray.Dataset
        The OOI glider DVL dataset with the depth-averaged eastward and
        northward current on the DVL timebase
    """
    surf = surf.sort_values("surface_time")
    start = surf["dive_start_time"].values.astype("datetime64[ns]")
    end = surf["surface_time"].values.astype("datetime64[ns]")
    dvl_time = dvl["time"].values.astype("datetime64[ns]")

    # Find the first surfacing at or after each DVL time, and check the time is in its segment
    idx = np.searchsorted(end, dvl_time, side="left")
    in_segment = idx < end.size
    in_segment[in_segment] = dvl_time[in_segment] >= start[idx[in_segment]]

    comment = (
        "The depth-averaged current over the dives between two surfacings, calculated from the "
        "displacement between the glider's first GPS fix and its dead-reckoned position at the "
        "surfacing, divided by the time underwater, plus the current used in the dead reckoning."
    )
    for name, column, direction in [
        ("depth_averaged_eastward_velocity", "current_east", "Eastward"),
        ("depth_averaged_northward_velocity", "current_north", "Northward"),
    ]:
        values = np.full(dvl_time.shape, np.nan)
        values[in_segment] = surf[column].values[idx[in_segment]]
        dvl[name] = (["time"], values)
        dvl[name].attrs = {
            "long_name": f"Depth-Averaged {direction} Sea Water Velocity",
            "units": "m s-1",
            "comment": comment,
        }

    return dvl
