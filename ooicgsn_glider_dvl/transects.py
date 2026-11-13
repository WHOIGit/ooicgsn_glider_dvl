"""
@author Andrew Reed
@brief Provides methods for splitting a glider deployment into transects along the edges of
    its survey box, positioning each profile along its transect, and flagging the profiles
    where the glider was off course
"""

import numpy as np
import pandas as pd
import xarray as xr

# Corners of the Pioneer-NES glider survey box (degrees north and east)
NORTH_LAT = 40.083332
SOUTH_LAT = 39.833332
WEST_LON = -71.166664
EAST_LON = -70.375000

# The edges of the Pioneer-NES survey box, each a straight line between two corners. The
# glider can fly an edge in either direction, so the order of the corners does not matter.
PIONEER_NES_TRANSECTS = pd.DataFrame({
    "transect_id": [1, 2, 3, 4],
    "transect_name": ["North_Edge", "West_Edge", "South_Edge", "East_Edge"],
    "start_lat": [NORTH_LAT, NORTH_LAT, SOUTH_LAT, SOUTH_LAT],
    "start_lon": [EAST_LON, WEST_LON, WEST_LON, EAST_LON],
    "end_lat": [NORTH_LAT, SOUTH_LAT, SOUTH_LAT, NORTH_LAT],
    "end_lon": [WEST_LON, WEST_LON, EAST_LON, EAST_LON],
})

# 4-point cardinal directions, in order of increasing bearing
CARDINAL_POINTS = np.array(["North", "East", "South", "West"])

# Kilometers per degree of latitude (mean Earth radius of 6371 km)
KM_PER_DEGREE = 111.195


def calculate_bearing(lat1, lon1, lat2, lon2):
    """Calculate the initial great-circle bearing from one position to another

    Parameters
    ----------
    lat1, lon1: float or array-like
        The starting position (degrees north and east)
    lat2, lon2: float or array-like
        The ending position (degrees north and east)

    Returns
    -------
    bearing: float or numpy.ndarray
        The bearing (degrees, 0-360) clockwise from north: 0 = north, 90 = east,
        180 = south, 270 = west. NaN where either position is missing.
    """
    lat1, lon1, lat2, lon2 = (
        np.radians(np.asarray(x, dtype=float)) for x in (lat1, lon1, lat2, lon2)
    )
    dlon = lon2 - lon1
    x = np.sin(dlon) * np.cos(lat2)
    y = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
    return np.degrees(np.arctan2(x, y)) % 360


def bearing_difference(bearing1, bearing2):
    """Calculate the smallest angle between two bearings

    Parameters
    ----------
    bearing1, bearing2: float or array-like
        Bearings (degrees clockwise from north)

    Returns
    -------
    difference: float or numpy.ndarray
        The absolute angle (degrees, 0-180) between the bearings, accounting for the
        wrap-around at north, e.g. 350 and 10 are 20 degrees apart. NaN where either
        bearing is missing.
    """
    diff = np.asarray(bearing2, dtype=float) - np.asarray(bearing1, dtype=float)
    return np.abs((diff + 180) % 360 - 180)


def profile_positions(ds: xr.Dataset, lat: str = "lat", lon: str = "lon") -> pd.DataFrame:
    """Summarize the start position, start time, and waypoint of each glider profile

    Parameters
    ----------
    ds: xarray.Dataset
        The glider DVL dataset with profile_id, time, and the latitude and longitude
        on the time dimension (as variables or coordinates). If the dataset has the
        glider's waypoint (waypoint_lat and waypoint_lon, from merge.add_waypoints),
        the waypoint of each profile is included.
    lat, lon: str
        The names of the latitude and longitude variables

    Returns
    -------
    profiles: pandas.DataFrame
        One row per profile, sorted by start time, with:
            - profile_id
            - time_first: the time of the profile's first measurement
            - lat_start, lon_start: the position at the profile's first measurement,
              which is used as the position of the profile
            - lat_mean, lon_mean: the mean position over the profile
            - waypoint_lat, waypoint_lon: if available, the waypoint the glider was
              heading to at the end of the profile. Missing and (0, 0) waypoints are NaN.
    """
    df = pd.DataFrame({
        "profile_id": ds["profile_id"].values,
        "lat": ds[lat].values,
        "lon": ds[lon].values,
        "time": ds["time"].values,
    })
    aggregations = {
        "time_first": ("time", "first"),
        "lat_start": ("lat", "first"),
        "lon_start": ("lon", "first"),
        "lat_mean": ("lat", "mean"),
        "lon_mean": ("lon", "mean"),
    }
    if "waypoint_lat" in ds and "waypoint_lon" in ds:
        wpt_lat = ds["waypoint_lat"].values.astype(float)
        wpt_lon = ds["waypoint_lon"].values.astype(float)
        unset = (wpt_lat == 0) & (wpt_lon == 0)
        df["waypoint_lat"] = np.where(unset, np.nan, wpt_lat)
        df["waypoint_lon"] = np.where(unset, np.nan, wpt_lon)
        aggregations["waypoint_lat"] = ("waypoint_lat", "last")
        aggregations["waypoint_lon"] = ("waypoint_lon", "last")

    df = df.sort_values("time", kind="stable")
    profiles = df.groupby("profile_id").agg(**aggregations)
    return profiles.sort_values("time_first").reset_index()


def assign_transects(
    profiles: pd.DataFrame,
    transects: pd.DataFrame = PIONEER_NES_TRANSECTS,
    min_coverage: float = 0.8,
    waypoint_tolerance: float = 0.5,
) -> pd.DataFrame:
    """Group the profiles into full-length transects along the survey box edges

    Each profile is positioned at its start (lat_start, lon_start), and matched to the
    transect (box edge) closest to it, measured as the shortest distance to the line
    segment between the transect's two ends. Distances are calculated on a local
    flat-earth projection, with longitude scaled by the cosine of the box's mean latitude
    so that a degree east and a degree north are the same length.

    The position of each profile on its transect is given by its along_distance, the
    distance (km) along the edge from the transect's start corner, and its
    cross_distance, the distance (km) from the edge. The along_distance is measured from
    the same corner whichever direction the glider flies, so it is a common horizontal
    axis for the cross sections of every pass along a transect. It is not limited to the
    ends of the transect: a profile beyond the start corner has a negative along_distance,
    and one beyond the end corner has an along_distance greater than the transect length.

    The glider can fly a transect in either direction, and can turn back partway along it
    when its waypoint changes. The principal direction of each profile is the direction
    along its transect (e.g. East or West for the North_Edge) closest to the bearing from
    the profile to the waypoint the glider was heading to. Where the waypoint is not
    available, the principal direction is taken from the glider's net travel along the
    transect instead: the bearing from the first to the last profile of the unbroken run
    of profiles on the transect (or from the previous profile, for a run of one profile).

    The profiles are then grouped into passes: an unbroken run of consecutive profiles on
    the same transect in the same principal direction. A pass ends when the glider moves
    to another transect, turns back along the transect toward a new waypoint, or leaves
    the box.

    Only full or nearly full-length passes along the box edges are kept as transects. The
    rest of the profiles are off-transect:
        - Profiles steering to a waypoint more than waypoint_tolerance from the box edges,
          such as in transit to or from the box. The glider flies the box by steering to
          waypoints on its edges, usually its corners.
        - Passes that cover less than min_coverage of the length of their transect, such
          as the short runs where the glider loops around the corner it is heading for,
          turns back partway along an edge, or is held in place by a strong current. The
          coverage is the distance along the transect between the pass's two farthest
          apart profiles (within the transect's ends), as a fraction of its length.
    Where the waypoint is not available (no waypoint columns, or a missing waypoint), the
    waypoint check is skipped. The defaults are strict, preferring to drop a pass that
    may be a usable transect over keeping one that is not.

    A pass is kept whole when a current pushes the glider off course, as long as the
    glider makes its way back to complete the transect. The profiles where it was off
    course are flagged by flag_deviations, so they can be dropped.

    Off-transect profiles have transect_id 0, transect_name and principal_direction
    "Off_Transect", transect_pass and transect_profile_num 0, and no position on a
    transect, so they are never deviations.

    The transects are numbered in time order on each box edge, so the third full-length
    pass along the north edge, in either direction, is pass 3 of the North_Edge.

    Parameters
    ----------
    profiles: pandas.DataFrame
        The profile positions from profile_positions, sorted by time. The
        waypoint_lat and waypoint_lon columns are optional.
    transects: pandas.DataFrame, Default = PIONEER_NES_TRANSECTS
        One row per transect with transect_id (1 or greater), transect_name, and the
        start_lat, start_lon, end_lat, end_lon of its two ends
    min_coverage: float, Default = 0.8
        The smallest fraction (0-1) of the transect's length a pass must cover to be kept.
        Set to 0 to keep passes of any length.
    waypoint_tolerance: float, Default = 0.5
        The farthest distance (km) of the glider's waypoint from a box edge for the
        profile to be on a transect

    Returns
    -------
    profiles: pandas.DataFrame
        The profiles with added columns:
            - transect_id, transect_name: of the transect, or 0 and "Off_Transect"
            - transect_distance: distance (km) from the profile to the nearest box edge,
              within its ends
            - transect_coverage: fraction of the nearest edge's length covered by the
              profile's pass, before the short passes are dropped. NaN for profiles
              dropped by the waypoint check.
            - transect_length: length (km) of the transect
            - along_distance: distance (km) along the transect from its start corner
            - cross_distance: distance (km) from the transect's line
            - transect_progress: distance (km) along the transect in the principal
              direction, from the corner the pass started at: equal to along_distance on
              a pass from the start corner to the end corner, and to transect_length
              minus along_distance on a pass the other way
            - transect_pass: the pass number along the transect
            - principal_direction: the direction along the transect toward the glider's
              waypoint (North, East, South, or West)
            - expected_bearing: the bearing (degrees) along the transect in the principal
              direction
            - transect_profile_num: the profile number along the transect, counting
              through all of its passes
            - profile_identifier: unique label of the transect, direction, and profile
              number, e.g. "T1_W_P0042" for the 42nd profile on transect 1, on a pass
              heading West, or "Off_Transect"
        The transect positions (transect_length through transect_progress) and the
        expected_bearing are NaN for off-transect profiles.
    """
    profiles = profiles.reset_index(drop=True)
    lat, lon = profiles["lat_start"].values, profiles["lon_start"].values

    geometry = _transect_geometry(lat, lon, transects)
    nearest_idx = np.argmin(geometry["distance"], axis=1)
    rows = np.arange(len(profiles))
    nearest = {
        name: geometry[name][rows, nearest_idx] for name in ("distance", "fraction", "cross")
    }
    nearest_id = transects["transect_id"].values[nearest_idx]
    length = geometry["length"][nearest_idx]
    profiles["transect_distance"] = nearest["distance"]

    # Keep the profiles steering to a waypoint on the box
    on_transect = np.ones(len(profiles), dtype=bool)
    has_waypoints = "waypoint_lat" in profiles and "waypoint_lon" in profiles
    if has_waypoints:
        wpt_lat = profiles["waypoint_lat"].values.astype(float)
        wpt_lon = profiles["waypoint_lon"].values.astype(float)
        waypoint_distance = _transect_geometry(wpt_lat, wpt_lon, transects)["distance"]
        on_transect &= ~(waypoint_distance.min(axis=1) > waypoint_tolerance)

    # Bearing along each profile's transect, from its start corner to its end corner
    transect_bearing = calculate_bearing(
        transects["start_lat"], transects["start_lon"], transects["end_lat"], transects["end_lon"]
    )[nearest_idx]

    # Fallback heading: the net travel over each unbroken run of profiles on a transect
    run = _runs(np.where(on_transect, nearest_id, 0))
    index = pd.Series(rows)
    first = index.groupby(run).transform("first").values
    last = index.groupby(run).transform("last").values
    start = np.where(first == last, np.maximum(first - 1, 0), first)
    heading = calculate_bearing(lat[start], lon[start], lat[last], lon[last])

    # Heading toward the waypoint, where the glider's waypoint is available
    if has_waypoints:
        to_waypoint = calculate_bearing(lat, lon, wpt_lat, wpt_lon)
        heading = np.where(np.isfinite(to_waypoint), to_waypoint, heading)

    # Principal direction: whichever way along the transect is closest to the heading
    reverse = bearing_difference(transect_bearing, heading) > 90
    expected_bearing = (transect_bearing + 180 * reverse) % 360
    direction = CARDINAL_POINTS[np.round(expected_bearing / 90).astype(int) % 4]

    # Drop the passes that cover less than min_coverage of their transect
    run = _runs(np.where(on_transect, nearest_id, 0), direction)
    within_ends = pd.Series(np.clip(nearest["fraction"], 0, 1)).groupby(run)
    coverage = (within_ends.transform("max") - within_ends.transform("min")).values
    profiles["transect_coverage"] = np.where(on_transect, coverage, np.nan)
    on_transect &= coverage >= min_coverage

    profiles["transect_id"] = np.where(on_transect, nearest_id, 0)
    profiles["transect_name"] = np.where(
        on_transect, transects["transect_name"].values[nearest_idx], "Off_Transect"
    )

    # Position on the transect, from its start corner and in the principal direction
    along = nearest["fraction"] * length
    profiles["transect_length"] = np.where(on_transect, length, np.nan)
    profiles["along_distance"] = np.where(on_transect, along, np.nan)
    profiles["cross_distance"] = np.where(on_transect, nearest["cross"], np.nan)
    profiles["transect_progress"] = np.where(
        on_transect, np.where(reverse, length - along, along), np.nan
    )

    profiles["principal_direction"] = np.where(on_transect, direction, "Off_Transect")
    profiles["expected_bearing"] = np.where(on_transect, expected_bearing, np.nan)

    # Number the passes in time order on each transect
    run = _runs(profiles["transect_id"].values, profiles["principal_direction"].values)
    passes = run.groupby(profiles["transect_id"]).rank(method="dense").astype(int)
    profiles["transect_pass"] = np.where(on_transect, passes, 0)

    profile_num = profiles.groupby("transect_id").cumcount() + 1
    profiles["transect_profile_num"] = np.where(on_transect, profile_num, 0)
    profiles["profile_identifier"] = [
        f"T{tid}_{direction[0]}_P{num:04d}" if tid > 0 else "Off_Transect"
        for tid, direction, num in zip(
            profiles["transect_id"],
            profiles["principal_direction"],
            profiles["transect_profile_num"],
        )
    ]
    return profiles


def _runs(*keys) -> pd.Series:
    """Number the unbroken runs of consecutive rows with the same values of all of the keys"""
    changed = np.zeros(len(keys[0]), dtype=bool)
    changed[0:1] = True
    for key in keys:
        key = np.asarray(key)
        changed[1:] |= key[1:] != key[:-1]
    return pd.Series(np.cumsum(changed))


def _transect_geometry(lat, lon, transects: pd.DataFrame) -> dict:
    """Position of each point (rows) relative to each transect (columns)

    Points and transect ends are projected so that degrees east and north are the same
    length, using the cosine of the box's mean latitude. Missing points are NaN.

    Returns a dict of:
        - distance: distance (km) to the nearest point on the transect, within its ends
        - fraction: position along the transect's line as a fraction of its length, 0 at
          the start corner and 1 at the end corner, and outside 0-1 beyond the corners
        - cross: perpendicular distance (km) from the transect's line
        - length: length (km) of each transect (one value per transect)
    """
    scale = np.cos(np.radians(transects[["start_lat", "end_lat"]].values.mean()))
    px = np.asarray(lon, dtype=float)[:, None] * scale
    py = np.asarray(lat, dtype=float)[:, None]
    x0 = transects["start_lon"].values * scale
    y0 = transects["start_lat"].values
    dx = transects["end_lon"].values * scale - x0
    dy = transects["end_lat"].values - y0

    # Project each point onto each transect's line
    seg_len_sq = dx**2 + dy**2
    fraction = ((px - x0) * dx + (py - y0) * dy) / np.where(seg_len_sq == 0, 1, seg_len_sq)
    cross = np.hypot(px - (x0 + fraction * dx), py - (y0 + fraction * dy))
    clipped = np.clip(fraction, 0, 1)
    distance = np.hypot(px - (x0 + clipped * dx), py - (y0 + clipped * dy))
    return {
        "distance": distance * KM_PER_DEGREE,
        "fraction": fraction,
        "cross": cross * KM_PER_DEGREE,
        "length": np.sqrt(seg_len_sq) * KM_PER_DEGREE,
    }


def identify_glider_transects(
    ds: xr.Dataset,
    transects: pd.DataFrame = PIONEER_NES_TRANSECTS,
    lat: str = "lat",
    lon: str = "lon",
    min_coverage: float = 0.8,
    waypoint_tolerance: float = 0.5,
) -> tuple[xr.Dataset, pd.DataFrame]:
    """Identify the transect that each glider profile is on and add it to the dataset

    Summarizes each profile's start position and waypoint (profile_positions), assigns
    the profiles flown along the survey box edges to transects (assign_transects), and
    maps the transect information back onto every measurement in the dataset through its
    profile_id. Only full or nearly full-length passes along the box edges are kept as
    transects. The rest of the profiles, such as in transit to the box or in short or
    partial passes, are marked off-transect. See assign_transects.

    Every measurement in a profile is given the profile's position, at its start. The
    along_distance is the horizontal axis for the cross sections of a transect.

    Parameters
    ----------
    ds: xarray.Dataset
        The glider DVL dataset with profile_id, time, and latitude and longitude on the
        time dimension
    transects: pandas.DataFrame, Default = PIONEER_NES_TRANSECTS
        The transects (box edges) to assign the profiles to. See assign_transects.
    lat, lon: str
        The names of the latitude and longitude variables
    min_coverage: float, Default = 0.8
        The smallest fraction (0-1) of the transect's length a pass must cover to be kept
    waypoint_tolerance: float, Default = 0.5
        The farthest distance (km) of the glider's waypoint from a box edge for the
        profile to be on a transect

    Returns
    -------
    ds: xarray.Dataset
        A copy of the dataset with transect_id, transect_name, principal_direction,
        transect_pass, profile_identifier, along_distance, and cross_distance on the
        time dimension
    profiles: pandas.DataFrame
        One row per profile with its position, transect, and pass, from assign_transects
    """
    profiles = assign_transects(
        profile_positions(ds, lat=lat, lon=lon),
        transects,
        min_coverage=min_coverage,
        waypoint_tolerance=waypoint_tolerance,
    )

    ds = _add_profile_variables(ds, profiles, {
        "transect_id": (-1, {
            "long_name": "Transect ID",
            "comment": "Number of the survey box edge the profile is on (0 = off transect)",
        }),
        "transect_name": ("Unknown", {
            "long_name": "Transect Name",
            "comment": "Name of the survey box edge the profile is on, or Off_Transect",
        }),
        "principal_direction": ("Unknown", {
            "long_name": "Principal Direction",
            "comment": "Direction along the transect toward the glider's waypoint",
        }),
        "transect_pass": (-1, {
            "long_name": "Transect Pass",
            "comment": "Number of the full-length pass, in time order (0 = off transect)",
        }),
        "profile_identifier": ("Unknown", {
            "long_name": "Profile Identifier",
            "comment": "Transect number, principal direction, and profile number on the transect",
        }),
        "along_distance": (np.nan, {
            "long_name": "Distance Along Transect",
            "units": "km",
            "comment": "Distance along the transect from its start corner, at the profile start",
        }),
        "cross_distance": (np.nan, {
            "long_name": "Distance From Transect",
            "units": "km",
            "comment": "Perpendicular distance from the transect's line, at the profile start",
        }),
    })
    return ds, profiles


def flag_deviations(
    profiles: pd.DataFrame,
    max_distance: float | None = 5,
    overshoot_tolerance: float = 0,
    backtrack_tolerance: float = 0,
) -> pd.DataFrame:
    """Flag the profiles on a transect where the glider was off course

    The cross sections of a transect are made on the distance along it, so the profiles
    that are flagged are those that do not add a new position on the transect's line:
        - off_edge: the profile is more than max_distance from the transect's line
          (cross_distance), e.g. where a current pushed the glider off the edge.
        - overshoot: the profile is beyond either corner of the transect, by more than
          overshoot_tolerance, e.g. where the glider passed its waypoint before turning.
        - backtrack: the profile is behind the farthest point along the transect, in the
          principal direction, already reached by an earlier profile on the pass, by
          more than backtrack_tolerance. This flags the profiles where the glider was
          moving the wrong way, e.g. pushed back by a current stronger than its speed,
          and the profiles on its way back up to where it had been, which would sample
          positions along the transect that the pass had already sampled. Only the
          profiles that are not off_edge or overshoot count toward the farthest point.
    A profile is a deviation if any of the three flags are set. Off-transect profiles are
    never deviations. The first profile of a pass is never a backtrack.

    Consecutive deviations on the same pass along a transect are grouped into numbered
    deviation segments, in time order. A segment ends at the first profile that is not a
    deviation, or at the end of the pass.

    Parameters
    ----------
    profiles: pandas.DataFrame
        The profiles with their transects and positions from assign_transects, sorted
        by time
    max_distance: float, Default = 5
        The distance (km) from the transect's line beyond which a profile is off_edge.
        Set to None to not flag profiles by their distance from the line.
    overshoot_tolerance: float, Default = 0
        The distance (km) beyond a corner of the transect at which a profile is overshoot
    backtrack_tolerance: float, Default = 0
        The distance (km) behind the farthest point reached on the pass at which a
        profile is backtrack

    Returns
    -------
    profiles: pandas.DataFrame
        The profiles with added columns off_edge, overshoot, backtrack, is_deviation (any
        of the three), and deviation_segment (the number of the deviation segment, or 0
        if not a deviation)
    """
    profiles = profiles.copy()
    on_transect = profiles["transect_id"] > 0
    along = profiles["along_distance"]
    length = profiles["transect_length"]

    off_edge = on_transect & False
    if max_distance is not None:
        off_edge = on_transect & (profiles["cross_distance"] > max_distance)
    overshoot = on_transect & (
        (along < -overshoot_tolerance) | (along > length + overshoot_tolerance)
    )

    # Farthest progress along the transect reached by the earlier profiles on each pass
    pass_key = [profiles["transect_id"], profiles["transect_pass"]]
    counts = on_transect & ~off_edge & ~overshoot
    progress = profiles["transect_progress"].where(counts, -np.inf)
    farthest = progress.groupby(pass_key).cummax().groupby(pass_key).shift()
    backtrack = counts & (profiles["transect_progress"] < farthest - backtrack_tolerance)

    profiles["off_edge"] = off_edge
    profiles["overshoot"] = overshoot
    profiles["backtrack"] = backtrack
    profiles["is_deviation"] = off_edge | overshoot | backtrack

    # A segment starts at a deviation unless the previous profile was a deviation on the same pass
    deviation = profiles["is_deviation"]
    same_pass = (profiles["transect_id"] == profiles["transect_id"].shift()) & (
        profiles["transect_pass"] == profiles["transect_pass"].shift()
    )
    starts = deviation & ~(deviation.shift(fill_value=False) & same_pass)
    profiles["deviation_segment"] = starts.cumsum().where(deviation, 0)
    return profiles


def summarize_deviations(profiles: pd.DataFrame) -> dict:
    """Summarize the deviations from the transects

    Parameters
    ----------
    profiles: pandas.DataFrame
        The profiles with deviations from flag_deviations

    Returns
    -------
    summary: dict
        - total_profiles: number of profiles
        - transect_profiles: number of profiles on transects
        - deviation_profiles: number of profiles on transects flagged as deviations
        - deviation_percentage: percent of the profiles on transects that are deviations
        - num_off_edge, num_overshoot, num_backtrack: number of profiles with each flag
        - num_deviation_segments: number of deviation segments
        - segment_details: pandas.DataFrame with one row per deviation segment: its
          first and last profile_id and number of profiles, its transect, pass, and
          principal direction, the number of its profiles with each flag, the start
          times of its first and last profiles, and the duration between them (0 for a
          single-profile segment)
    """
    n_transect = int((profiles["transect_id"] > 0).sum())
    n_deviations = int(profiles["is_deviation"].sum())

    deviations = profiles[profiles["deviation_segment"] > 0]
    segments = deviations.groupby("deviation_segment").agg(
        start_profile=("profile_id", "first"),
        end_profile=("profile_id", "last"),
        num_profiles=("profile_id", "count"),
        transect_id=("transect_id", "first"),
        transect_name=("transect_name", "first"),
        transect_pass=("transect_pass", "first"),
        principal_direction=("principal_direction", "first"),
        num_off_edge=("off_edge", "sum"),
        num_overshoot=("overshoot", "sum"),
        num_backtrack=("backtrack", "sum"),
        start_time=("time_first", "first"),
        end_time=("time_first", "last"),
    )
    segments = segments.rename_axis("segment_id").reset_index()
    segments["duration"] = segments["end_time"] - segments["start_time"]

    return {
        "total_profiles": len(profiles),
        "transect_profiles": n_transect,
        "deviation_profiles": n_deviations,
        "deviation_percentage": 100 * n_deviations / n_transect if n_transect else np.nan,
        "num_off_edge": int(profiles["off_edge"].sum()),
        "num_overshoot": int(profiles["overshoot"].sum()),
        "num_backtrack": int(profiles["backtrack"].sum()),
        "num_deviation_segments": len(segments),
        "segment_details": segments,
    }


def add_deviations_to_dataset(ds: xr.Dataset, profiles: pd.DataFrame) -> xr.Dataset:
    """Add the deviations from the transects to the glider dataset

    Maps the deviation flags for each profile onto every measurement in the dataset
    through its profile_id.

    Parameters
    ----------
    ds: xarray.Dataset
        The glider DVL dataset with profile_id on the time dimension
    profiles: pandas.DataFrame
        The profiles with deviations from flag_deviations

    Returns
    -------
    ds: xarray.Dataset
        A copy of the dataset with off_edge, overshoot, backtrack, is_deviation, and
        deviation_segment on the time dimension
    """
    return _add_profile_variables(ds, profiles, {
        "off_edge": (False, {
            "long_name": "Off Edge",
            "comment": "True where the profile start is too far from the transect's line",
        }),
        "overshoot": (False, {
            "long_name": "Overshoot",
            "comment": "True where the profile start is beyond a corner of the transect",
        }),
        "backtrack": (False, {
            "long_name": "Backtrack",
            "comment": "True where the profile start is behind the farthest point of the pass",
        }),
        "is_deviation": (False, {
            "long_name": "Is Deviation",
            "comment": "True where the profile is off_edge, overshoot, or backtrack",
        }),
        "deviation_segment": (0, {
            "long_name": "Deviation Segment",
            "comment": "Number of the run of consecutive deviating profiles (0 = no deviation)",
        }),
    })


def _add_profile_variables(ds: xr.Dataset, profiles: pd.DataFrame, variables: dict) -> xr.Dataset:
    """Add per-profile values to a copy of the dataset on the time dimension

    variables maps each column of profiles to (fill value for profiles not in the table,
    attributes)
    """
    ds = ds.copy()
    lookup = profiles.set_index("profile_id")
    profile_id = ds["profile_id"].values
    for name, (fill, attrs) in variables.items():
        values = lookup[name].reindex(profile_id, fill_value=fill).values
        ds[name] = (["time"], values, attrs)
    return ds


def find_latitude_segment_crossings(
    time,
    lat,
    lon,
    lat0: float = NORTH_LAT,
    lon_min: float = WEST_LON,
    lon_max: float = EAST_LON,
) -> list[dict]:
    """Find where a trajectory crosses a line of constant latitude between two longitudes

    Each consecutive pair of trajectory points is checked for a crossing of lat0. The
    crossing longitude and time are linearly interpolated between the two points, and
    only crossings between lon_min and lon_max are kept. By default the line is the north
    edge of the Pioneer-NES survey box, e.g. to find where a warm core ring trajectory
    crosses it.

    A point exactly on lat0 is counted as being north of it, so a trajectory that
    touches the line counts as one crossing, not two. Pairs with a missing position
    are skipped.

    Parameters
    ----------
    time: array-like of datetime64, pandas.Timestamp, or float
        The time of each trajectory point
    lat: array-like of float
        Latitude (degrees north) of each trajectory point
    lon: array-like of float
        Longitude (degrees east, -180 to 180) of each trajectory point
    lat0: float, Default = NORTH_LAT
        The latitude of the line
    lon_min: float, Default = WEST_LON
        The western end of the line
    lon_max: float, Default = EAST_LON
        The eastern end of the line

    Returns
    -------
    crossings: list[dict]
        One dict per crossing, in trajectory order, with:
            - time: interpolated time of the crossing
            - lon: interpolated longitude of the crossing
            - direction: "northward" or "southward"
            - index: the crossing is between trajectory points index and index + 1
    """
    time = np.asarray(time)
    if time.dtype == object:
        time = pd.to_datetime(time).values
    lat = np.asarray(lat, dtype=float)
    lon = np.asarray(lon, dtype=float)

    lat1, lat2 = lat[:-1], lat[1:]
    lon1, lon2 = lon[:-1], lon[1:]
    valid = np.isfinite(lat1) & np.isfinite(lat2) & np.isfinite(lon1) & np.isfinite(lon2)
    crosses = valid & ((lat1 < lat0) != (lat2 < lat0))
    idx = np.flatnonzero(crosses)

    frac = (lat0 - lat1[idx]) / (lat2[idx] - lat1[idx])
    lon_cross = lon1[idx] + frac * (lon2[idx] - lon1[idx])
    keep = (lon_cross >= lon_min) & (lon_cross <= lon_max)
    idx, frac, lon_cross = idx[keep], frac[keep], lon_cross[keep]

    t1, t2 = time[idx], time[idx + 1]
    if np.issubdtype(time.dtype, np.datetime64):
        dt = (t2 - t1).astype("timedelta64[ns]").astype(np.int64)
        t_cross = t1 + np.round(frac * dt).astype(np.int64).astype("timedelta64[ns]")
    else:
        t_cross = t1 + frac * (t2 - t1)

    return [
        {
            "time": t,
            "lon": float(x),
            "direction": "northward" if lat[i + 1] > lat[i] else "southward",
            "index": int(i),
        }
        for t, x, i in zip(t_cross, lon_cross, idx)
    ]
