"""
@author Andrew Reed
@brief Provides methods for constructing absolute water velocity profiles from the glider DVL by
    least squares, constrained by the glider's velocity through the water from the flight
    model and referenced to the depth-averaged current or bottom tracking, and for comparing
    the profiles with the depth-averaged current
"""

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
import xarray as xr

from ooicgsn_glider_dvl.binning import VELOCITY_VARIABLES, add_profile_start, good_velocity_cells

# The glider's velocity through the water calculated by the flight model (flight.py)
GLIDER_VELOCITY_VARIABLES = (
    "glider_eastward_velocity_through_water",
    "glider_northward_velocity_through_water",
)

# The depth-averaged current from the glider's displacement (currents.py)
DEPTH_AVERAGED_VARIABLES = (
    "depth_averaged_eastward_velocity",
    "depth_averaged_northward_velocity",
)


def invert_profiles(
    ds: xr.Dataset,
    bin_size: float = 2,
    max_depth: float | None = None,
    steady: np.ndarray | None = None,
    velocity_variables: tuple[str, str] = VELOCITY_VARIABLES,
    glider_velocity_variables: tuple[str, str] = GLIDER_VELOCITY_VARIABLES,
    depth_averaged_variables: tuple[str, str] = DEPTH_AVERAGED_VARIABLES,
    depth: str = "depth_from_pressure",
    flight_weight: float = 1,
    dac_weight: float = 1000,
    bottom_track_variables: tuple[str, str] | None = None,
    bottom_track_weight: float = 10,
    use_bottom_track: np.ndarray | None = None,
    smoothness: float = 0.1,
    max_segments: int | None = 1,
    lat: str = "lat",
    lon: str = "lon",
    **good_cell_kwargs,
) -> tuple[xr.Dataset, xr.Dataset]:
    """Construct absolute water velocity profiles from the DVL, flight model, and a reference

    Without bottom tracking, the DVL measures the water velocity relative to the glider,
    which moves over the ground with an unknown velocity. For each glider profile, the
    water velocity profile v(z), on depth bins of bin_size meters, and the glider's
    velocity over the ground g(t) of each ensemble are solved for together by least
    squares (as for a lowered ADCP; Visbeck, 2002; Todd et al., 2017), from:

        - the ADCP equations, one for each good cell measured during steady flight:
          v(z_cell) - g(t) = v_relative(t, cell)
        - the flight model equations, one for each ensemble: the glider moves over the
          ground with its velocity through the water from the flight model plus the
          current at its depth, g(t) - v(z_glider(t)) = v_glider_through_water(t), with
          weight flight_weight relative to each ADCP equation
        - the depth-averaged current equation: the mean of v at the glider's depth over
          all of the profile's measurements equals the depth-averaged current measured
          from the glider's displacement, i.e. the current averaged over the time the
          glider was underwater. It has weight dac_weight, so it is effectively exact.
          Set dac_weight to 0 to leave it out.
        - the bottom-track equations (with bottom_track_variables), one for each ensemble
          with bottom tracking during steady flight: the glider's velocity over the
          ground is the opposite of the bottom-track velocity, g(t) = -v_bottom_track(t),
          with weight bottom_track_weight relative to each ADCP equation
        - smoothness equations: a small penalty (smoothness) on the difference between
          adjacent depth bins, which regularizes the depth bins that the other equations
          don't constrain

    The ADCP and flight model equations are unchanged by adding the same constant to v
    and g, so they only determine the profile up to a constant. The absolute level comes
    from the reference: the depth-averaged current (from the GPS), bottom tracking, or
    both. Profiles without either are not solved. Referencing to bottom tracking and
    leaving out the depth-averaged current gives profiles that are independent of the
    depth-averaged current, to check it with (see compare_depth_averaged_current), and
    use_bottom_track can hold some of the bottom tracking out to check the profiles with.

    The differences between the cells of each ensemble determine the vertical shear
    independently of the glider's velocity. The flight model equations link the shear
    of the different ensembles together and to the depth of the glider, but an error in
    the flight model then acts as a false shear over the distance between the glider and
    the cells, which adds up over the depth of the profile. Where there are many good
    cells in each ensemble, as on the shelf, the shear from the cells dominates. Where
    there are only a few (e.g. in deep water with few scatterers), the ADCP equations
    split the depth bins into separate segments that only the flight model links
    together, and the profiles are less reliable. The number of these segments is
    returned for each profile (n_segments), and profiles with more than max_segments
    are not solved.

    The depth bins above the shallowest cell, where the glider is but there are no
    cells, take the velocity of the nearest depth bin with cells. The depth-averaged
    current is measured over all of the dives between two surfacings, which can be more
    than one glider profile, and over the climbs, where the ADCP is off. Errors in the
    depth-averaged current, or a current that changes over that time, offset the whole
    profile. Bottom tracking measures the glider's velocity over the ground directly,
    independently of the flight model and the depth-averaged current, so it can be used
    to check the profiles and the solved glider velocities (see bottom_track.py), or as
    the reference.

    The eastward and northward components are solved separately.

    Parameters
    ----------
    ds: xarray.Dataset
        The glider DVL dataset with profile_id, the depth variable, the velocity, glider
        velocity, and depth-averaged current variables, and bin_depths
    bin_size: float, Default = 2
        The height (m) of the depth bins, which are the same as binning.bin_profiles
    max_depth: float, Optional
        The deepest depth (m) of the depth bins. Default is the deepest ADCP cell.
    steady: numpy.ndarray[bool], Optional
        The measurements where the glider is in steady flight. Default is
        flight.calibration_mask(ds).
    velocity_variables: tuple[str, str]
        The eastward and northward water velocities relative to the glider (time, bin)
    glider_velocity_variables: tuple[str, str]
        The eastward and northward glider velocities through the water from the flight
        model (time)
    depth_averaged_variables: tuple[str, str]
        The eastward and northward depth-averaged current (time)
    depth: str, Default = "depth_from_pressure"
        The depth (m) of the glider
    flight_weight: float, Default = 1
        The weight of each flight model equation relative to each ADCP equation, e.g.
        the ratio of the uncertainty of an ADCP cell to that of the flight model
    dac_weight: float, Default = 1000
        The weight of the depth-averaged current equation relative to each ADCP equation.
        0 leaves it out.
    bottom_track_variables: tuple[str, str], Optional
        The eastward and northward bottom-track velocities, the velocity of the bottom
        relative to the glider (time), e.g. ("bottom_track_eastward_velocity",
        "bottom_track_northward_velocity") from bottom_track.add_bottom_track. Default is
        not to use bottom tracking.
    bottom_track_weight: float, Default = 10
        The weight of each bottom-track equation relative to each ADCP equation.
        Bottom-track velocities are much less noisy than those of a single ADCP cell.
    use_bottom_track: numpy.ndarray[bool], Optional
        The measurements whose bottom tracking to use, e.g. to hold the rest out to check
        the profiles with. Default is all of them.
    smoothness: float, Default = 0.1
        The weight of the difference between adjacent depth bins relative to each ADCP
        equation
    max_segments: int or None, Default = 1
        The most segments the ADCP equations can split a profile into for it to be
        solved. With the default, only the profiles where the cells link every depth bin
        together are solved; checked against bottom tracking, the others have errors of
        tens of cm s-1. None solves every profile.
    lat, lon: str
        The names of the latitude and longitude variables
    good_cell_kwargs:
        Passed to binning.good_velocity_cells to select the good cells, e.g.
        qc_variable, bad_flags, or remove_bottom_hits

    Returns
    -------
    profiles: xarray.Dataset
        The profiles on the profile and depth dimensions, with the same depth bins and
        per-profile variables as binning.bin_profiles, and for each velocity variable:
            - {variable}: the absolute water velocity (m s-1)
            - {variable}_residual: the rms residual (m s-1) of the ADCP equations of the
              cells in the depth bin, a measure of the uncertainty
        with the number of ADCP cells in each depth bin (n_cells) and the number of
        segments the ADCP equations split each profile into (n_segments; 0 where there
        are no equations) and the number of bottom-track equations of each profile
        (n_bottom_track). The velocities are NaN in depth bins without any ADCP cells,
        and in profiles without steady flight or a reference, or split into more than
        max_segments segments.
    ground: xarray.Dataset
        On the time dimension of the dataset:
            - glider_eastward_ground_velocity, glider_northward_ground_velocity: the solved
              glider velocity over the ground (m s-1), NaN where the ensemble isn't in the
              equations
            - eastward_velocity_at_glider, northward_velocity_at_glider: the water
              velocity (m s-1) of the solved profile at the glider's depth, for every
              measurement of the solved profiles (the velocity the depth-averaged current
              equation averages)
    """
    if steady is None:
        from ooicgsn_glider_dvl.flight import calibration_mask

        steady = calibration_mask(ds)
    reference = " and ".join(
        name
        for name, used in [
            ("depth-averaged current", dac_weight > 0),
            ("bottom tracking", bottom_track_variables is not None),
        ]
        if used
    )
    if not reference:
        raise ValueError(
            "Use the depth-averaged current, bottom tracking, or both as the reference"
        )

    profile_id = ds["profile_id"].values
    glider_depth = ds[depth].values.astype(float)
    cell_depths = ds["bin_depths"].transpose("time", "bin").values.astype(float)
    if max_depth is None:
        max_depth = np.nanmax(cell_depths)
    n_bins = int(np.floor(max_depth / bin_size)) + 1
    edges = np.arange(n_bins + 1) * bin_size
    glider_bin = np.floor(glider_depth / bin_size)
    cell_bin = np.floor(cell_depths / bin_size)

    # The ADCP equations: the good cells measured during steady flight
    glider_velocity = np.column_stack([ds[v].values for v in glider_velocity_variables])
    in_range = np.isfinite(glider_bin) & (glider_bin >= 0) & (glider_bin < n_bins)
    good = good_velocity_cells(ds, velocity_variables, **good_cell_kwargs)
    good &= (steady & np.isfinite(glider_velocity).all(axis=1) & in_range)[:, None]
    good &= np.isfinite(cell_bin) & (cell_bin < n_bins)
    relative = np.stack(
        [ds[v].transpose("time", "bin").values.astype(float) for v in velocity_variables],
        axis=-1,
    )
    dac = np.column_stack([ds[v].values for v in depth_averaged_variables])

    # The bottom-track equations: the ensembles with bottom tracking during steady flight
    if bottom_track_variables is not None:
        bottom_track = np.column_stack([ds[v].values for v in bottom_track_variables])
        has_bottom_track = np.isfinite(bottom_track).all(axis=1)
        has_bottom_track &= steady & np.isfinite(glider_velocity).all(axis=1) & in_range
        if use_bottom_track is not None:
            has_bottom_track &= use_bottom_track
    else:
        bottom_track = np.full((ds.sizes["time"], 2), np.nan)
        has_bottom_track = np.zeros(ds.sizes["time"], dtype=bool)

    profile_ids = np.unique(profile_id)
    shape = (profile_ids.size, n_bins, 2)
    velocity = np.full(shape, np.nan)
    residual = np.full(shape, np.nan)
    count = np.zeros(shape[:2], dtype=int)
    segments = np.zeros(profile_ids.size, dtype=int)
    n_bottom_track = np.zeros(profile_ids.size, dtype=int)
    ground = np.full((ds.sizes["time"], 2), np.nan)
    at_glider_velocity = np.full((ds.sizes["time"], 2), np.nan)

    order = np.argsort(profile_id, kind="stable")
    splits = np.flatnonzero(np.diff(profile_id[order])) + 1
    for p, idx in enumerate(np.split(order, splits)):
        rows, cols = np.nonzero(good[idx])
        bt_rows = np.flatnonzero(has_bottom_track[idx])
        use_dac = dac_weight > 0 and np.isfinite(dac[idx]).all(axis=1).any()
        if rows.size == 0 or not (use_dac or bt_rows.size):
            continue
        profile_dac = np.nanmean(dac[idx], axis=0) if use_dac else np.zeros(2)

        # The unknowns are the velocities in the depth bins with ADCP cells, and the
        # glider velocity of each ensemble with cells or bottom tracking. A glider depth
        # bin without cells takes the velocity of the nearest depth bin with cells.
        k_cell = cell_bin[idx][rows, cols].astype(int)
        observed = np.unique(k_cell)
        ensembles = np.union1d(rows, bt_rows)
        e = np.searchsorted(ensembles, rows)
        bt_e = np.searchsorted(ensembles, bt_rows)
        cell = np.searchsorted(observed, k_cell)
        at_glider = _nearest(observed, glider_bin[idx][ensembles])
        in_profile = np.flatnonzero(in_range[idx])
        mean = _nearest(observed, glider_bin[idx][in_profile])

        # The segments of depth bins linked by the cells of the same ensembles
        n = observed.size
        graph = coo_matrix((np.ones(e.size), (cell, n + e)), shape=(n + ensembles.size,) * 2)
        labels = connected_components(graph, directed=False)[1]
        segments[p] = np.unique(labels[:n]).size
        if max_segments is not None and segments[p] > max_segments:
            continue

        u, g = _solve_profile(
            cell,
            e,
            at_glider,
            mean,
            relative[idx][rows, cols],
            glider_velocity[idx][ensembles],
            profile_dac,
            bt_e,
            -bottom_track[idx][bt_rows],
            n,
            flight_weight,
            dac_weight if use_dac else 0,
            bottom_track_weight,
            smoothness,
        )
        velocity[p, observed] = u
        ground[idx[ensembles]] = g
        at_glider_velocity[idx[in_profile]] = u[mean]
        n_bottom_track[p] = bt_rows.size

        # The rms residual and number of cells in each depth bin
        misfit = relative[idx][rows, cols] - (u[cell] - g[e])
        count[p, observed] = np.bincount(cell, minlength=n)
        for c in range(2):
            squares = np.bincount(cell, weights=misfit[:, c] ** 2, minlength=n)
            residual[p, observed, c] = np.sqrt(squares / count[p, observed])

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
    for c, name in enumerate(velocity_variables):
        long_name = ds[name].attrs.get("long_name", name)
        profiles[name] = (
            ("profile", "depth"),
            velocity[..., c],
            {
                "long_name": f"Absolute {long_name}",
                "units": "m s-1",
                "comment": (
                    f"{long_name} from the DVL by least squares, constrained by the glider's "
                    f"velocity through the water from the flight model and referenced to the "
                    f"{reference}"
                ),
            },
        )
        profiles[f"{name}_residual"] = (
            ("profile", "depth"),
            residual[..., c],
            {
                "long_name": f"RMS Residual of {long_name}",
                "units": "m s-1",
                "comment": "RMS residual of the ADCP equations of the cells in the depth bin",
            },
        )
    profiles["n_cells"] = (
        ("profile", "depth"),
        count,
        {"long_name": "Number of ADCP Cells", "comment": "Number of ADCP cells in the depth bin"},
    )
    profiles["n_segments"] = (
        "profile",
        segments,
        {
            "long_name": "Number of Profile Segments",
            "comment": (
                "Number of separate segments of depth bins that the ADCP cells of the same "
                "ensembles link together, which only the flight model links to each other"
            ),
        },
    )
    profiles["n_bottom_track"] = (
        "profile",
        n_bottom_track,
        {
            "long_name": "Number of Bottom-Track Equations",
            "comment": "Number of ensembles with bottom tracking used in the profile",
        },
    )
    profiles = add_profile_start(profiles, ds, lat, lon)
    profiles.attrs = {
        "comment": (
            "Absolute water velocity profiles from the glider DVL by least squares, "
            "constrained by the glider's velocity through the water from the flight model "
            f"and referenced to the {reference}, on {bin_size} m depth bins"
        ),
        "bin_size": bin_size,
        "flight_weight": flight_weight,
        "dac_weight": dac_weight,
        "bottom_track_weight": bottom_track_weight if bottom_track_variables else 0,
        "reference": reference,
        "smoothness": smoothness,
        "max_segments": "none" if max_segments is None else max_segments,
    }

    ground_velocity = xr.Dataset(coords={"time": ds["time"]})
    for c, direction in enumerate(["eastward", "northward"]):
        ground_velocity[f"glider_{direction}_ground_velocity"] = (
            "time",
            ground[:, c],
            {
                "long_name": f"Glider {direction.capitalize()} Velocity Over the Ground",
                "units": "m s-1",
                "comment": "Glider velocity over the ground solved for with the profiles",
            },
        )
        ground_velocity[f"{direction}_velocity_at_glider"] = (
            "time",
            at_glider_velocity[:, c],
            {
                "long_name": f"{direction.capitalize()} Water Velocity at the Glider",
                "units": "m s-1",
                "comment": "Water velocity of the solved profile at the glider's depth",
            },
        )
    ground_velocity.attrs = {"reference": profiles.attrs["reference"]}
    return profiles, ground_velocity


def compare_depth_averaged_current(
    ground: xr.Dataset,
    surfacings: pd.DataFrame,
    min_coverage: float = 0.9,
    ds: xr.Dataset | None = None,
    steady: np.ndarray | None = None,
) -> pd.DataFrame:
    """Compare the depth-averaged current with the time-mean of the profiles at the glider

    The depth-averaged current from the glider's displacement (currents.py) is the
    current averaged over the time the glider was underwater between two surfacings, at
    the depths the glider was at. The same average of the solved profiles is the mean of
    their water velocity at the glider's depth over every DVL measurement between the
    surfacings. For profiles referenced to the depth-averaged current, the two agree by
    construction. For profiles referenced only to bottom tracking (invert_profiles with
    dac_weight=0), they are independent, so the difference checks the depth-averaged
    current (and the profiles).

    The ADCP is off during the climbs, so the mean of the profiles is only over the
    dives. It is the time-mean over the whole time underwater if the current at each
    depth is the same on the dive and the climb, and the glider spends the same time at
    each depth on both.

    The depth-averaged current also depends on the glider's own estimate of its speed
    through the water (m_speed): it is the GPS displacement minus the dead-reckoned
    displacement through the water, over the time underwater, so an error in m_speed
    goes straight into it. With ds, the mean difference between the flight model's and
    the glider's horizontal velocity through the water during steady flight is also
    returned, and the depth-averaged current recalculated with the flight model's speed
    instead of m_speed. The mean is time-weighted, over the measurements of ds in the
    segment, on the time base of ds: with the glider dataset (merge.glider_dataset), it is
    over the dives and the climbs, as the depth-averaged current is; with the DVL dataset,
    over the dives only. The flight model is only valid in steady flight, so the mean over
    the steady flight is taken to be the mean over the whole segment.

    Parameters
    ----------
    ground: xarray.Dataset
        The ground velocity output of invert_profiles, with eastward_velocity_at_glider and
        northward_velocity_at_glider on the time dimension
    surfacings: pandas.DataFrame
        The depth-averaged current at each surfacing from currents.surfacing_currents,
        with dive_start_time, surface_time, current_east, and current_north
    min_coverage: float, Default = 0.9
        The smallest fraction of the DVL measurements between the surfacings that must
        have a profile velocity at the glider for the surfacing to be compared
    ds: xarray.Dataset, Optional
        The glider dataset (merge.glider_dataset) or the glider DVL dataset, on any time
        base, with vehicle_heading, dead_reckoning_speed (flight.add_flight_variables), and
        the flight model glider velocities (flight.solve_flight_model), to compare the
        speeds
    steady: numpy.ndarray[bool], Optional
        The measurements of ds where the glider is in steady flight, to compare the
        speeds. Default is flight.calibration_mask(ds).

    Returns
    -------
    comparison: pandas.DataFrame
        One row per surfacing with enough coverage, with the dive start and surfacing
        times, the number of DVL measurements and the fraction with a profile velocity
        (coverage), the time-mean of the profiles at the glider (profile_east/north), the
        depth-averaged current (dac_east/north), and their difference (profile minus
        depth-averaged current, difference_east/north), in m s-1. With ds, also the mean
        flight model minus dead-reckoning velocity through the water
        (flight_minus_dead_reckoning_east/north), the fraction of the time of the segment
        in ds in steady flight (steady_fraction), and the depth-averaged current
        recalculated with the flight model's speed (dac_flight_east/north).
    """
    surfacings = surfacings.sort_values("surface_time").reset_index(drop=True)
    start = pd.to_datetime(surfacings["dive_start_time"]).values.astype("datetime64[ns]")
    end = pd.to_datetime(surfacings["surface_time"]).values.astype("datetime64[ns]")

    velocity = pd.DataFrame(
        {
            "interval": _segment(ground["time"].values, start, end),
            "east": ground["eastward_velocity_at_glider"].values,
            "north": ground["northward_velocity_at_glider"].values,
        }
    )
    velocity = velocity[velocity["interval"] >= 0]
    grouped = velocity.groupby("interval")
    stats = (
        grouped[["east", "north"]]
        .mean()
        .rename(columns={"east": "profile_east", "north": "profile_north"})
    )
    stats["n_measurements"] = grouped.size()
    stats["coverage"] = grouped["east"].count() / stats["n_measurements"]

    comparison = surfacings.loc[stats.index, ["dive_start_time", "surface_time"]].join(stats)
    comparison["dac_east"] = surfacings.loc[stats.index, "current_east"]
    comparison["dac_north"] = surfacings.loc[stats.index, "current_north"]
    comparison["difference_east"] = comparison["profile_east"] - comparison["dac_east"]
    comparison["difference_north"] = comparison["profile_north"] - comparison["dac_north"]
    if ds is not None:
        if steady is None:
            from ooicgsn_glider_dvl.flight import calibration_mask

            steady = calibration_mask(ds)
        # The flight model minus the glider's own velocity through the water, along its
        # heading, time-weighted over the steady flight of each segment
        seconds = ds["time"].values.astype("datetime64[ns]").astype("int64") / 1e9
        # Each measurement stands for the time around it, but not across a gap in the data
        weight = np.gradient(seconds) if seconds.size > 1 else np.ones(seconds.shape)
        weight = (
            np.minimum(weight, 3 * np.median(np.diff(seconds))) if seconds.size > 1 else weight
        )
        heading = np.deg2rad(ds["vehicle_heading"].values)
        speed = np.hypot(
            ds["glider_eastward_velocity_through_water"].values,
            ds["glider_northward_velocity_through_water"].values,
        )
        difference = speed - ds["dead_reckoning_speed"].values
        use = steady & np.isfinite(difference) & np.isfinite(heading)
        segment = _segment(ds["time"].values, start, end)
        frame = pd.DataFrame(
            {
                "interval": segment,
                "weight": weight,
                "steady_weight": np.where(use, weight, 0),
                "east": np.where(use, weight * difference * np.sin(heading), 0),
                "north": np.where(use, weight * difference * np.cos(heading), 0),
            }
        )
        sums = frame[frame["interval"] >= 0].groupby("interval").sum()
        steady_time = sums["steady_weight"].where(sums["steady_weight"] > 0)
        for direction in ["east", "north"]:
            mean = (sums[direction] / steady_time).reindex(comparison.index)
            comparison[f"flight_minus_dead_reckoning_{direction}"] = mean
            comparison[f"dac_flight_{direction}"] = comparison[f"dac_{direction}"] - mean
        comparison["steady_fraction"] = (sums["steady_weight"] / sums["weight"]).reindex(
            comparison.index
        )
    comparison = comparison[comparison["coverage"] >= min_coverage]
    return comparison.reset_index(drop=True)


def _segment(time, start, end):
    """The index of the segment (start to end) each time is in, or -1 if it is in none"""
    time = np.asarray(time).astype("datetime64[ns]")
    interval = np.searchsorted(end, time, side="left")
    inside = interval < end.size
    inside[inside] = time[inside] >= start[interval[inside]]
    return np.where(inside, interval, -1)


def _solve_profile(
    cell,
    e,
    at_glider,
    mean,
    relative,
    through_water,
    dac,
    bt_e,
    over_ground,
    n,
    flight_weight,
    dac_weight,
    bottom_track_weight,
    smoothness,
):
    """Solve the least squares equations of one profile

    The unknowns are the water velocities u in the n depth bins with ADCP cells, and the
    glider velocity g of each ensemble with cells. For each ADCP equation, cell and e are
    its depth bin and ensemble; at_glider is the depth bin of the glider in each ensemble,
    and mean the depth bins of the glider in every measurement of the profile. bt_e are
    the ensembles with bottom tracking, and over_ground their glider velocity over the
    ground from bottom tracking. The glider
    velocities only appear in the equations of their own ensemble, so they are eliminated
    from the normal equations (by the Schur complement) to solve for u, then calculated
    from u. Returns u on the (bin, component) dimensions and g on the (ensemble,
    component) dimensions.
    """
    n_ensembles = through_water.shape[0]
    weight = flight_weight**2

    # Normal equations of the ADCP equations u(cell) - g(e) = relative and the flight model
    # equations g(e) - u(at_glider(e)) = through_water(e)
    uu = np.zeros((n, n))
    uu[np.diag_indices(n)] = np.bincount(cell, minlength=n) + weight * np.bincount(
        at_glider, minlength=n
    )
    ug = np.zeros((n, n_ensembles))
    np.add.at(ug, (cell, e), -1)
    np.add.at(ug, (at_glider, np.arange(n_ensembles)), -weight)
    gg = np.bincount(e, minlength=n_ensembles) + weight

    bu = np.zeros((n, 2))
    np.add.at(bu, cell, relative)
    np.add.at(bu, at_glider, -weight * through_water)
    bg = np.zeros((n_ensembles, 2))
    np.add.at(bg, e, -relative)
    bg += weight * through_water

    # The bottom-track equations g(bt_e) = over_ground
    gg = gg.astype(float)
    gg[bt_e] += bottom_track_weight**2
    bg[bt_e] += bottom_track_weight**2 * over_ground

    # The depth-averaged current: the mean of u at the glider's depths
    mean = np.bincount(mean, minlength=n) / mean.size
    uu += dac_weight**2 * np.outer(mean, mean)
    bu += dac_weight**2 * np.outer(mean, dac)

    # Smoothness: the difference between adjacent depth bins
    diff = np.diff(np.eye(n), axis=0)
    uu += smoothness**2 * diff.T @ diff

    # Eliminate the glider velocities and solve for the water velocities
    scaled = ug / gg
    u = np.linalg.lstsq(uu - scaled @ ug.T, bu - scaled @ bg, rcond=None)[0]
    g = (bg - ug.T @ u) / gg[:, None]
    return u, g


def _nearest(observed, bins):
    """Index of the nearest observed depth bin to each depth bin"""
    if observed.size == 1:
        return np.zeros(bins.shape, dtype=int)
    above = np.clip(np.searchsorted(observed, bins), 1, observed.size - 1)
    below = above - 1
    closer_below = (bins - observed[below]) <= (observed[above] - bins)
    return np.where(closer_below, below, above)
