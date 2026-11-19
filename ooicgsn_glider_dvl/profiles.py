"""
@author Andrew Reed
@brief Provides methods for splitting glider ADCP datasets into separate profiles, labeling the
    glider's dives, climbs, and surfacings, and linking the ADCP profiles to the science profiles
"""

import numpy as np
import pandas as pd
import xarray as xr

from ooicgsn_glider_dvl.utils import unix_time


def identify_profiles(
    ds: xr.Dataset, min_depth_change: float = 2, min_time_gap: float = 4
) -> list[np.ndarray]:
    """Identify the start of each profile dive.

    The glider ADCP is only on during the dive portion
    of each glider profile, sampling every ~2 seconds. We
    identify where the change in depth between consecutive
    measurements is greater than min_depth_change and the
    time between them is greater than min_time_gap, indicating
    the ADCP was off between them and a new dive has started.

    Parameters
    ----------
    ds: xarray.Dataset
        A dataset of the glider ADCP data with the parameter
        'depth_from_pressure'
    min_depth_change: float, Default = 2
        The change in depth, in meters, between consecutive
        measurements that starts a new dive
    min_time_gap: float, Default = 4
        The time, in seconds, between consecutive measurements
        that starts a new dive

    Returns
    -------
    inflection_idx: list[numpy.array[int]]
        The indices of the data points in each profile dive
    """
    # Take the absolute difference in depth
    dz = np.abs(np.diff(ds["depth_from_pressure"].values))

    # Get the time id and the time between measurements
    time_ = unix_time(ds["time"].values)
    dt = np.diff(time_)

    # Identify where the depth and time both jump and get
    # the associated times with those inflection points
    new_dive = (dz > min_depth_change) & (dt > min_time_gap)
    inflection_times = np.concatenate([time_[:1], time_[:-1][new_dive], time_[-1:]])

    # Each profile spans (inflection_times[n], inflection_times[n+1]], so the
    # profile of each point is the inflection interval it falls in
    interval = np.searchsorted(inflection_times, time_, side="left") - 1
    valid = np.flatnonzero(interval >= 0)
    if valid.size == 0:
        return []

    # Group the indices by profile, dropping empty intervals
    order = np.argsort(interval[valid], kind="stable")
    sorted_interval = interval[valid][order]
    starts = np.flatnonzero(np.diff(sorted_interval)) + 1
    inflection_idx = np.split(valid[order], starts)

    return inflection_idx


def get_profile_ids(
    ds: xr.Dataset, min_depth_change: float = 2, min_time_gap: float = 4
) -> np.ndarray[int]:
    """Create profile ids for the glider ADCP dataset

    Parameters
    ----------
    ds: xarray.Dataset
        A dataset of the glider ADCP data with the parameter
        'depth_from_pressure'
    min_depth_change: float, Default = 2
        The change in depth, in meters, between consecutive
        measurements that starts a new dive
    min_time_gap: float, Default = 4
        The time, in seconds, between consecutive measurements
        that starts a new dive

    Returns
    -------
    profiles: numpy.array[int]
        An array the length of the input dataset with the
        profile id added based on the inflection indices
    """

    # First, get the inflection indices
    profile_idx = identify_profiles(ds, min_depth_change, min_time_gap)

    # Next, number each profile sequentially
    profiles = np.zeros(ds["time"].shape, dtype=np.int32)
    for n, idx in enumerate(profile_idx):
        profiles[idx] = n

    return profiles


def add_profiles(
    ds: xr.Dataset, min_depth_change: float = 2, min_time_gap: float = 4
) -> xr.Dataset:
    """Add a sequential id to each glider profile dive

    The glider ADCP is only on during the dive portion
    of each glider profile, sampling every ~2 seconds. We
    identify where the change in depth between consecutive
    measurements is greater than min_depth_change (2 meters)
    and the time between them is greater than min_time_gap
    (4 seconds), indicating the start of a new dive. Each
    sampling point is then matched with the start and stop times
    of an individual profile, and the profile id is added to
    the dataset.

    Parameters
    ----------
    ds: xarray.Dataset
        A dataset of the glider ADCP data with the parameter
        'depth_from_pressure'
    min_depth_change: float, Default = 2
        The change in depth, in meters, between consecutive
        measurements that starts a new dive
    min_time_gap: float, Default = 4
        The time, in seconds, between consecutive measurements
        that starts a new dive

    Returns
    -------
    ds: xarray.Dataset
        The input dataset of the glider ADCP data with each
        time point matched with its profile dive id

    """
    # Get the profiles
    profiles = get_profile_ids(ds, min_depth_change, min_time_gap)

    # Add to the dataset
    ds["profile_id"] = (["time"], profiles)
    ds["profile_id"].attrs = {
        "comment": (
            "The id of each profile in the dataset counting sequentially from the first profile."
        ),
        "long_name": "Profile ID",
        "ancillary_variables": "depth_from_pressure",
    }

    return ds


# Values of the phase variable added by add_phases
PHASES = {"unlabeled": 0, "dive": 1, "climb": 2, "surface": 3}


def science_profiles(gdac: xr.Dataset) -> pd.DataFrame:
    """List the dive and climb profiles of the IOOS GDAC science dataset

    The GDAC dataset is a contiguous ragged array of profiles: the observations of each
    profile are consecutive along the time dimension, with the number of observations in
    rowSize. Each dive and each climb is a separate profile, and the glider is at the
    surface, or turning at a shallow inflection, in the gaps between them. A profile is a
    dive if the CTD pressure at its end is greater than at its start, and a climb
    otherwise.

    Parameters
    ----------
    gdac: xarray.Dataset
        The OOI IOOS GDAC glider science dataset, with profile_id and rowSize on the
        profile dimension and pressure (dbar) on the time dimension

    Returns
    -------
    profiles: pandas.DataFrame
        One row per profile with the science_profile_id, the start and end times, the
        phase (dive or climb), the number of CTD measurements, and the minimum and maximum
        CTD pressure (dbar)
    """
    edges = np.concatenate([[0], np.cumsum(gdac["rowSize"].values)])
    n = edges.size - 1
    time = gdac["time"].values
    profile = np.repeat(np.arange(n), np.diff(edges))
    by_profile = pd.Series(gdac["pressure"].values.astype(float)).groupby(profile)

    def per_profile(stat):
        return getattr(by_profile, stat)().reindex(range(n)).values

    return pd.DataFrame(
        {
            "science_profile_id": gdac["profile_id"].values.astype(int),
            "start_time": time[edges[:-1]],
            "end_time": time[edges[1:] - 1],
            "phase": np.where(per_profile("last") > per_profile("first"), "dive", "climb"),
            "n_ctd": per_profile("count"),
            "min_pressure": per_profile("min"),
            "max_pressure": per_profile("max"),
        }
    )


def add_phases(ds: xr.Dataset, profiles: pd.DataFrame, min_surface_time: float = 60) -> xr.Dataset:
    """Label each time with its science profile and whether the glider is diving, climbing,
    or at the surface

    Each time from the start to the end of a science profile (see science_profiles) is
    given its science_profile_id and phase (dive or climb). The glider is at the surface
    in the gaps between profiles that last at least min_surface_time: the climbs end a few
    meters below the surface and the dives that follow start at the surface. The shorter
    gaps (about 1 s) are the shallow inflections between a climb and a dive that don't
    surface, and are left unlabeled, as are the times before the first or after the last
    profile.

    Parameters
    ----------
    ds: xarray.Dataset
        A dataset with a time dimension, e.g. the glider dataset from merge.glider_dataset
    profiles: pandas.DataFrame
        The science profiles from science_profiles
    min_surface_time: float, Default = 60
        The shortest gap between profiles, in seconds, that is a surfacing

    Returns
    -------
    ds: xarray.Dataset
        The dataset with science_profile_id (NaN outside of a profile) and phase on the
        time dimension
    """
    profiles = profiles.sort_values("start_time")
    start = profiles["start_time"].values.astype("datetime64[ns]")
    end = profiles["end_time"].values.astype("datetime64[ns]")
    time = ds["time"].values.astype("datetime64[ns]")

    # The last profile starting at or before each time, and whether the time is in it
    idx = np.searchsorted(start, time, side="right") - 1
    after_first = idx >= 0
    inside = after_first.copy()
    inside[after_first] = time[after_first] <= end[idx[after_first]]

    profile_id = np.full(time.shape, np.nan)
    profile_id[inside] = profiles["science_profile_id"].values[idx[inside]]
    phase = np.full(time.shape, PHASES["unlabeled"], dtype=np.int8)
    phase[inside] = np.where(
        profiles["phase"].values[idx[inside]] == "dive", PHASES["dive"], PHASES["climb"]
    )

    # The times in a gap between two profiles that lasts at least min_surface_time
    between = after_first & ~inside & (idx < start.size - 1)
    gap = np.zeros(time.shape)
    gap[between] = (start[idx[between] + 1] - end[idx[between]]) / np.timedelta64(1, "s")
    phase[between & (gap >= min_surface_time)] = PHASES["surface"]

    ds["science_profile_id"] = (
        "time",
        profile_id,
        {
            "long_name": "Science Profile ID",
            "comment": (
                "The profile_id of the dive or climb profile in the IOOS GDAC science dataset, "
                "NaN outside of a profile"
            ),
        },
    )
    ds["phase"] = (
        "time",
        phase,
        {
            "long_name": "Glider Phase",
            "comment": (
                "Whether the glider is diving or climbing (in a science profile), or at the "
                f"surface (in a gap between profiles of at least {min_surface_time:g} s). "
                "The shallow inflections between a climb and a dive that don't surface, and "
                "the times before the first or after the last profile, are unlabeled."
            ),
            "flag_values": np.array(list(PHASES.values()), dtype=np.int8),
            "flag_meanings": " ".join(PHASES),
        },
    )
    return ds


def link_profiles(ds: xr.Dataset, profiles: pd.DataFrame) -> pd.DataFrame:
    """Link each ADCP profile to the science profile it overlaps the most in time

    The ADCP profiles (see add_profiles) start and end at slightly different times than
    the science profiles: the ADCP turns on a few seconds before a dive that starts at a
    shallow inflection, and a couple of minutes after a dive that starts at the surface.
    Each ADCP profile is linked to the science profile it overlaps the most. That is
    normally a dive, but the ADCP occasionally records a few measurements near the end of
    a climb.

    Parameters
    ----------
    ds: xarray.Dataset
        The glider ADCP dataset with profile_id on the time dimension
    profiles: pandas.DataFrame
        The science profiles from science_profiles

    Returns
    -------
    links: pandas.DataFrame
        One row per ADCP profile, indexed by its profile_id, with its start and end times
        and number of measurements, the linked science_profile_id and its phase, the
        fraction of the ADCP profile's duration that overlaps it, and the time from the
        start of the science profile to the start of the ADCP profile (s)
    """
    profiles = profiles.sort_values("start_time")
    adcp = (
        pd.DataFrame({"time": ds["time"].values, "profile_id": ds["profile_id"].values})
        .groupby("profile_id")["time"]
        .agg(start_time="min", end_time="max", n_measurements="size")
    )
    start = profiles["start_time"].values.astype("datetime64[ns]")
    end = profiles["end_time"].values.astype("datetime64[ns]")
    a_start = adcp["start_time"].values.astype("datetime64[ns]")
    a_end = adcp["end_time"].values.astype("datetime64[ns]")

    # The science profiles that end after each ADCP profile starts and start before it ends
    first = np.searchsorted(end, a_start, side="left")
    last = np.searchsorted(start, a_end, side="right")
    best = np.zeros(len(adcp), dtype=int)
    overlap = np.zeros(len(adcp))
    for n in range(len(adcp)):
        k = np.arange(min(first[n], start.size - 1), max(last[n], first[n] + 1))
        k = k[k < start.size]
        ov = np.minimum(end[k], a_end[n]) - np.maximum(start[k], a_start[n])
        ov = ov / np.timedelta64(1, "s")
        best[n], overlap[n] = k[np.argmax(ov)], ov.max()

    duration = (a_end - a_start) / np.timedelta64(1, "s")
    with np.errstate(invalid="ignore", divide="ignore"):
        fraction = np.where(duration > 0, np.clip(overlap, 0, None) / duration, overlap >= 0)
    return adcp.assign(
        science_profile_id=profiles["science_profile_id"].values[best],
        phase=profiles["phase"].values[best],
        overlap_fraction=fraction.astype(float),
        start_lag=(a_start - start[best]) / np.timedelta64(1, "s"),
    )


def add_profile_links(
    adcp: xr.Dataset, glider: xr.Dataset, links: pd.DataFrame
) -> tuple[xr.Dataset, xr.Dataset]:
    """Add the linked profiles to the ADCP and glider datasets

    Each ADCP measurement is given the science_profile_id and phase of the science profile
    its ADCP profile is linked to (see link_profiles). Each glider measurement in a
    science profile is given the profile_id of the ADCP profile linked to it, including in
    the parts of the science profile before the ADCP turned on or after it turned off. If
    more than one ADCP profile is linked to a science profile, the one that overlaps it
    the most is used.

    Parameters
    ----------
    adcp: xarray.Dataset
        The glider ADCP dataset with profile_id on the time dimension
    glider: xarray.Dataset
        The glider dataset with science_profile_id on the time dimension (see add_phases)
    links: pandas.DataFrame
        The links from link_profiles

    Returns
    -------
    adcp, glider: xarray.Dataset
        The ADCP dataset with science_profile_id and phase, and the glider dataset with
        adcp_profile_id (NaN where no ADCP profile is linked), on the time dimension
    """
    adcp_profile = adcp["profile_id"].values
    adcp["science_profile_id"] = (
        "time",
        links["science_profile_id"].reindex(adcp_profile).values.astype(float),
        {
            "long_name": "Science Profile ID",
            "comment": (
                "The profile_id of the profile in the IOOS GDAC science dataset that the ADCP "
                "profile is linked to: the one it overlaps the most in time"
            ),
        },
    )
    phase = links["phase"].map(PHASES).reindex(adcp_profile)
    adcp["phase"] = (
        "time",
        phase.fillna(PHASES["unlabeled"]).values.astype(np.int8),
        {
            "long_name": "Glider Phase",
            "comment": "Whether the science profile linked to the ADCP profile is a dive or climb",
            "flag_values": np.array(list(PHASES.values()), dtype=np.int8),
            "flag_meanings": " ".join(PHASES),
        },
    )

    # The ADCP profile linked to each science profile: the one that overlaps it the most
    primary = links.reset_index().sort_values("overlap_fraction", ascending=False)
    primary = primary.drop_duplicates("science_profile_id").set_index("science_profile_id")
    science = pd.Series(glider["science_profile_id"].values)
    glider["adcp_profile_id"] = (
        "time",
        science.map(primary["profile_id"]).values.astype(float),
        {
            "long_name": "ADCP Profile ID",
            "comment": (
                "The profile_id of the ADCP profile linked to the science profile (see "
                "science_profile_id), NaN where no ADCP profile is linked"
            ),
        },
    )
    return adcp, glider
