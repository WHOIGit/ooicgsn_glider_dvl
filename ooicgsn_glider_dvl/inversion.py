"""
@author Andrew Reed
@brief Provides methods for constructing absolute water velocity profiles from the glider DVL by
    least squares, constrained by the glider's velocity through the water from the flight
    model and referenced to the depth-averaged current
"""

import numpy as np
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
    smoothness: float = 0.1,
    max_segments: int | None = 1,
    lat: str = "lat",
    lon: str = "lon",
    **good_cell_kwargs,
) -> tuple[xr.Dataset, xr.Dataset]:
    """Construct absolute water velocity profiles from the DVL, flight model, and depth-averaged current

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
        - smoothness equations: a small penalty (smoothness) on the difference between
          adjacent depth bins, which regularizes the depth bins that the other equations
          don't constrain

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
    to check the profiles and the solved glider velocities (see bottom_track.py).

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
        The weight of the depth-averaged current equation relative to each ADCP equation
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
        are no equations). The velocities are NaN in depth bins without any ADCP cells,
        and in profiles without steady flight or a depth-averaged current, or split into
        more than max_segments segments.
    ground: xarray.Dataset
        The solved glider velocity over the ground, glider_eastward_ground_velocity and
        glider_northward_ground_velocity (m s-1), on the time dimension of the dataset,
        which is NaN where the ensemble isn't in the equations
    """
    if steady is None:
        from ooicgsn_glider_dvl.flight import calibration_mask

        steady = calibration_mask(ds)

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

    profile_ids = np.unique(profile_id)
    shape = (profile_ids.size, n_bins, 2)
    velocity = np.full(shape, np.nan)
    residual = np.full(shape, np.nan)
    count = np.zeros(shape[:2], dtype=int)
    segments = np.zeros(profile_ids.size, dtype=int)
    ground = np.full((ds.sizes["time"], 2), np.nan)

    order = np.argsort(profile_id, kind="stable")
    splits = np.flatnonzero(np.diff(profile_id[order])) + 1
    for p, idx in enumerate(np.split(order, splits)):
        rows, cols = np.nonzero(good[idx])
        if rows.size == 0 or not np.isfinite(dac[idx]).all(axis=1).any():
            continue
        profile_dac = np.nanmean(dac[idx], axis=0)

        # The unknowns are the velocities in the depth bins with ADCP cells, and the
        # glider velocity of each ensemble with cells. A glider depth bin without cells
        # takes the velocity of the nearest depth bin with cells.
        k_cell = cell_bin[idx][rows, cols].astype(int)
        observed = np.unique(k_cell)
        ensembles, e = np.unique(rows, return_inverse=True)
        cell = np.searchsorted(observed, k_cell)
        at_glider = _nearest(observed, glider_bin[idx][ensembles])
        mean = _nearest(observed, glider_bin[idx][in_range[idx]])

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
            n,
            flight_weight,
            dac_weight,
            smoothness,
        )
        velocity[p, observed] = u
        ground[idx[ensembles]] = g

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
                    "velocity through the water from the flight model and referenced to the "
                    "depth-averaged current"
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
    profiles = add_profile_start(profiles, ds, lat, lon)
    profiles.attrs = {
        "comment": (
            "Absolute water velocity profiles from the glider DVL by least squares, "
            "constrained by the glider's velocity through the water from the flight model "
            f"and referenced to the depth-averaged current, on {bin_size} m depth bins"
        ),
        "bin_size": bin_size,
        "flight_weight": flight_weight,
        "dac_weight": dac_weight,
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
    return profiles, ground_velocity


def _solve_profile(
    cell,
    e,
    at_glider,
    mean,
    relative,
    through_water,
    dac,
    n,
    flight_weight,
    dac_weight,
    smoothness,
):
    """Solve the least squares equations of one profile

    The unknowns are the water velocities u in the n depth bins with ADCP cells, and the
    glider velocity g of each ensemble with cells. For each ADCP equation, cell and e are
    its depth bin and ensemble; at_glider is the depth bin of the glider in each ensemble,
    and mean the depth bins of the glider in every measurement of the profile. The glider
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
