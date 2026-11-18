"""
@author Andrew Reed
@brief Provides methods for modeling the glider's velocity through the water with the steady-state
    Slocum glider flight model (gliderflight; Merckelbach et al., 2019), calibrated against the
    glider's depth rate, and for comparing it with the velocity measured by the DVL
"""

import gliderflight
import numpy as np
import pandas as pd
import xarray as xr

from ooicgsn_glider_dvl.binning import good_velocity_cells
from ooicgsn_glider_dvl.merge import interp_1d

# Glider engineering variables needed by the flight model. m_de_oil_vol is the buoyancy change
# of deep (1000 m) gliders and m_ballast_pumped that of shallow (200 m) gliders.
FLIGHT_VARIABLES = ["time", "m_pitch", "m_heading", "m_de_oil_vol", "m_ballast_pumped"]

# Flight model output added to the dataset by solve_flight_model
FLIGHT_MODEL_VARIABLES = (
    "glider_speed_through_water",
    "angle_of_attack",
    "glider_eastward_velocity_through_water",
    "glider_northward_velocity_through_water",
    "glider_upward_velocity_through_water",
    "upward_water_velocity",
)


def add_flight_variables(
    ds: xr.Dataset,
    eng: pd.DataFrame,
    declination: float,
    buoyancy: str = "m_de_oil_vol",
    max_gap: float | None = 60,
) -> xr.Dataset:
    """Add the glider's pitch, heading, and buoyancy change to the glider DVL dataset

    The flight model needs the glider's pitch and buoyancy change, and its heading to
    turn the modeled speed into eastward and northward velocities. These are taken from
    the glider engineering data and linearly interpolated to the DVL times, each from
    its own non-missing values. DVL times inside a gap in a variable longer than
    max_gap are NaN.

    The glider's heading (m_heading) is relative to magnetic north. It is rotated by the
    magnetic declination to relative to true north, the same rotation OOI applies to the
    DVL water velocities (see utils.magnetic_declination).

    Parameters
    ----------
    ds: xarray.Dataset
        The glider DVL dataset
    eng: pandas.DataFrame
        The glider engineering data with the time, m_pitch (rad), m_heading (rad), and
        buoyancy (cc) columns, e.g. read with pd.read_csv(eng_file,
        usecols=FLIGHT_VARIABLES)
    declination: float
        The magnetic declination (degrees, positive east of north)
    buoyancy: str, Default = "m_de_oil_vol"
        The buoyancy change variable: m_de_oil_vol for deep gliders or
        m_ballast_pumped for shallow gliders
    max_gap: float or None, Default = 60
        The longest gap between engineering data points, in seconds, to interpolate
        across. None interpolates across all gaps.

    Returns
    -------
    ds: xarray.Dataset
        The dataset with vehicle_pitch (degrees), vehicle_heading (degrees true), and
        buoyancy_change (cc) on the time dimension
    """
    eng = eng.assign(time=pd.to_datetime(eng["time"])).sort_values("time")
    eng = eng.drop_duplicates(subset="time")
    x = eng["time"].values.astype("datetime64[ns]").astype("int64").astype(float)
    xi = ds["time"].values.astype("datetime64[ns]").astype("int64").astype(float)
    max_gap_ns = None if max_gap is None else max_gap * 1e9

    def interp(values):
        return interp_1d(xi, x, np.asarray(values, dtype=float), max_gap_ns)

    pitch = np.rad2deg(interp(eng["m_pitch"]))

    # Unwrap the heading so it interpolates across north correctly
    heading = eng["m_heading"].to_numpy(dtype=float, copy=True)
    valid = np.isfinite(heading)
    heading[valid] = np.unwrap(heading[valid])
    heading = (np.rad2deg(interp(heading)) + declination) % 360

    ds["vehicle_pitch"] = (
        "time",
        pitch,
        {
            "long_name": "Glider Pitch",
            "units": "degrees",
            "comment": "Glider pitch (m_pitch), positive nose up",
            "source_sensor": "m_pitch",
        },
    )
    ds["vehicle_heading"] = (
        "time",
        heading,
        {
            "long_name": "Glider Heading",
            "units": "degrees",
            "comment": (
                "Glider heading (m_heading) rotated from relative to magnetic north to relative "
                f"to true north by the magnetic declination of {declination:.2f} degrees"
            ),
            "source_sensor": "m_heading",
        },
    )
    ds["buoyancy_change"] = (
        "time",
        interp(eng[buoyancy]),
        {
            "long_name": "Buoyancy Change",
            "units": "cm3",
            "comment": "Volume of the glider's buoyancy change relative to neutral",
            "source_sensor": buoyancy,
        },
    )
    return ds


def calibration_mask(
    ds: xr.Dataset,
    min_depth: float = 10,
    turn_margin: float = 5,
    min_pitch: float = 15,
    max_buoyancy_rate: float = 0.5,
    depth: str = "depth_from_pressure",
) -> np.ndarray:
    """Find the measurements where the glider is in steady flight

    The steady-state flight model assumes that the forces on the glider are in balance,
    which is not the case while the glider accelerates: near the surface, near the
    bottom of a dive where the glider pumps oil and changes its pitch to turn around,
    and while it changes its buoyancy. Only the steady flight is used to calibrate the
    model or to compare it with the DVL.

    Parameters
    ----------
    ds: xarray.Dataset
        The glider DVL dataset with profile_id, the depth variable, vehicle_pitch, and
        buoyancy_change (see add_flight_variables)
    min_depth: float, Default = 10
        Exclude the measurements shallower than this depth (m)
    turn_margin: float, Default = 5
        Exclude the measurements within this distance (m) of the deepest depth of their
        profile, where the glider turns around
    min_pitch: float, Default = 15
        Exclude the measurements with a pitch (degrees) of less than this, up or down
    max_buoyancy_rate: float, Default = 0.5
        Exclude the measurements where the buoyancy change is changing faster than this
        rate (cm3 s-1)
    depth: str, Default = "depth_from_pressure"
        The depth (m) of the glider

    Returns
    -------
    steady: numpy.ndarray[bool]
        Whether the glider is in steady flight, on the time dimension
    """
    z = ds[depth].values
    deepest = pd.Series(z).groupby(ds["profile_id"].values).transform("max").to_numpy()
    seconds = ds["time"].values.astype("datetime64[ns]").astype("int64") / 1e9
    with np.errstate(invalid="ignore"):
        buoyancy_rate = np.abs(np.gradient(ds["buoyancy_change"].values, seconds))
        steady = (
            (z >= min_depth)
            & (z <= deepest - turn_margin)
            & (np.abs(ds["vehicle_pitch"].values) >= min_pitch)
            & (buoyancy_rate <= max_buoyancy_rate)
        )
    return steady


def dvl_glider_velocity(
    ds: xr.Dataset,
    cell: int = 0,
    velocities: tuple[str, str, str] = (
        "eastward_seawater_velocity",
        "northward_seawater_velocity",
        "upward_seawater_velocity",
    ),
    **good_cell_kwargs,
) -> xr.Dataset:
    """Add the glider's velocity through the water measured by the DVL

    Without bottom tracking, the DVL measures the water velocity relative to the
    glider. In the cell closest to the glider (about 3 m below it), the water moves
    with nearly the same current as at the glider, so the opposite of the relative
    velocity is the glider's velocity through the water. This is an independent
    measurement of what the flight model calculates. The difference includes the
    current shear across the distance to the cell.

    Parameters
    ----------
    ds: xarray.Dataset
        The glider DVL dataset with vehicle_heading (see add_flight_variables)
    cell: int, Default = 0
        The index of the ADCP cell to use, where 0 is the closest to the glider
    velocities: tuple[str, str, str]
        The eastward, northward, and upward water velocities relative to the glider
    good_cell_kwargs:
        Passed to binning.good_velocity_cells to select the good cells, e.g.
        qc_variable or bad_flags

    Returns
    -------
    ds: xarray.Dataset
        The dataset with dvl_glider_eastward_velocity, dvl_glider_northward_velocity,
        dvl_glider_upward_velocity, and dvl_glider_forward_speed (the horizontal
        velocity in the direction of the glider's heading) on the time dimension, which
        are NaN where the cell is bad
    """
    good = good_velocity_cells(ds, velocities[:2], **good_cell_kwargs)[:, cell]
    components = {}
    for name, direction in zip(velocities, ["eastward", "northward", "upward"]):
        values = -ds[name].transpose("time", "bin").values[:, cell].astype(float)
        components[direction] = np.where(good, values, np.nan)
        ds[f"dvl_glider_{direction}_velocity"] = (
            "time",
            components[direction],
            {
                "long_name": f"DVL {direction.capitalize()} Glider Velocity Through the Water",
                "units": "m s-1",
                "comment": (
                    f"The opposite of the {name} relative to the glider in ADCP cell "
                    f"{cell + 1}, which is the glider's velocity through the water"
                ),
            },
        )
    heading = np.deg2rad(ds["vehicle_heading"].values)
    ds["dvl_glider_forward_speed"] = (
        "time",
        components["eastward"] * np.sin(heading) + components["northward"] * np.cos(heading),
        {
            "long_name": "DVL Glider Horizontal Speed Through the Water",
            "units": "m s-1",
            "comment": (
                "Horizontal glider velocity through the water measured by the DVL in the "
                "direction of the glider's heading"
            ),
        },
    )
    return ds


def calibrate_flight_model(
    ds: xr.Dataset,
    mass: float,
    parameters: tuple[str, ...] = ("Vg", "Cd0"),
    steady: np.ndarray | None = None,
    constraints: tuple[str, ...] = ("dhdt",),
    weights: tuple[float, ...] | None = None,
    window: str | None = None,
    min_points: int = 1000,
    rho0: float = 1025,
    initial: dict | None = None,
    **coefficients,
) -> pd.DataFrame:
    """Calibrate the steady-state glider flight model

    The flight model calculates the glider's speed through the water from the balance
    of its buoyancy, weight, lift, and drag, given its pitch, buoyancy change, and the
    density of the water. Of the model parameters, the glider's volume (Vg) and the
    parasitic drag coefficient (Cd0) depend on the ballasting and the state of the
    glider, so they are calibrated for each deployment: the parameters are adjusted to
    minimize the mean squared difference between the modeled vertical velocity of the
    glider and its depth rate, calculated from the pressure. The glider's mass must be
    known, since only the ratio of mass to volume can be calibrated from the depth rate.

    The drag of the glider can increase during a deployment as it gets biofouled. With
    window, the model is also calibrated separately for each time window (e.g. "7D"
    for each week), so solve_flight_model can follow the changes.

    Only the steady flight is used to calibrate the model (see calibration_mask).

    Parameters
    ----------
    ds: xarray.Dataset
        The glider DVL dataset with time, ctd_pressure (dbar), ctd_density (in situ,
        kg m-3), and the flight variables (see add_flight_variables)
    mass: float
        The glider's mass (kg), as weighed before the deployment
    parameters: tuple[str], Default = ("Vg", "Cd0")
        The model parameters to calibrate, e.g. add "epsilon" (the glider's
        compressibility) for deep dives, or "ah" (the hull lift coefficient) with the
        u_relative constraint
    steady: numpy.ndarray[bool], Optional
        The measurements to calibrate with. Default is calibration_mask(ds).
    constraints: tuple[str], Default = ("dhdt",)
        The measured velocities to calibrate against: "dhdt" for the depth rate,
        "u_relative" for the DVL horizontal speed through the water
        (dvl_glider_forward_speed; see dvl_glider_velocity), or both. With the default,
        the DVL measurements are kept independent of the model to validate it.
    weights: tuple[float], Optional
        The relative weights of the constraints. Default is equal weights.
    window: str, Optional
        A pandas frequency (e.g. "7D") to also calibrate the model for each time window
    min_points: int, Default = 1000
        The fewest steady flight measurements to calibrate a time window with. Windows
        with fewer use the deployment calibration.
    rho0: float, Default = 1025
        The reference density (kg m-3) used to convert pressure to depth
    initial: dict, Optional
        The initial values of the calibrated parameters. Default is Vg = mass / rho0 and
        Cd0 = 0.15.
    coefficients:
        Other model coefficients to set, e.g. ah, Cd1, or epsilon. The others take the
        gliderflight defaults.

    Returns
    -------
    calibration: pandas.DataFrame
        One row for the whole deployment ("deployment") and one for each time window,
        with the window start and end, the number of measurements calibrated with, the
        calibrated parameters, and rms_dhdt, the rms difference (m s-1) between the
        modeled vertical velocity and the depth rate
    """
    if steady is None:
        steady = calibration_mask(ds)
    initial = {"Vg": mass / rho0, "Cd0": 0.15, **(initial or {})}
    # The initial values of the parameters that are not calibrated stay fixed
    coefficients = {**{k: v for k, v in initial.items() if k not in parameters}, **coefficients}
    data, valid = _model_input(ds, u_relative="u_relative" in constraints)
    use = steady[valid]
    time = ds["time"].values[valid]

    # Calibrate on the whole deployment, then on each time window starting from it
    windows = [("deployment", np.ones(time.shape, dtype=bool))]
    if window is not None:
        for start, idx in pd.Series(np.arange(time.size), index=time).groupby(
            pd.Grouper(freq=window)
        ):
            in_window = np.zeros(time.shape, dtype=bool)
            in_window[idx.to_numpy()] = True
            windows.append((start, in_window))

    rows = []
    values = {name: initial.get(name) for name in parameters}
    for label, in_window in windows:
        n_points = int((use & in_window).sum())
        row = {
            "window": label,
            "start": time[in_window].min() if in_window.any() else pd.NaT,
            "end": time[in_window].max() if in_window.any() else pd.NaT,
            "n_points": n_points,
        }
        if n_points < min_points:
            rows.append({**row, **dict.fromkeys(parameters, np.nan), "rms_dhdt": np.nan})
            continue
        subset = {k: (v[in_window] if v is not None else None) for k, v in data.items()}
        model = gliderflight.SteadyStateCalibrate(rho0=rho0)
        model.define(mg=mass, **_floats({**coefficients, **values}))
        model.set_input_data(**subset)
        model.set_mask(~use[in_window])
        result = {
            k: float(v)
            for k, v in model.calibrate(
                *parameters, constraints=constraints, weights=weights
            ).items()
        }
        if label == "deployment":
            values = result
        model.define(**result)
        model.solve(model.input_data)
        error = model.modelresult.w - model.input_data["dhdt"]
        rms = float(np.sqrt(np.nanmean(error[use[in_window]] ** 2)))
        rows.append({**row, **result, "rms_dhdt": rms})

    calibration = pd.DataFrame(rows)
    calibration.attrs = {"mass": mass, "rho0": rho0, **coefficients}
    return calibration


def solve_flight_model(ds: xr.Dataset, calibration: pd.DataFrame) -> xr.Dataset:
    """Calculate the glider's velocity through the water with the calibrated flight model

    The steady-state flight model is solved with the calibrated parameters for the whole
    deployment, then with those of each calibrated time window over its time window.
    The glider's horizontal speed through the water is in the direction of its heading
    (vehicle_heading), so the model neglects sideslip.

    Parameters
    ----------
    ds: xarray.Dataset
        The glider DVL dataset, as for calibrate_flight_model
    calibration: pandas.DataFrame
        The calibration from calibrate_flight_model

    Returns
    -------
    ds: xarray.Dataset
        The dataset with the FLIGHT_MODEL_VARIABLES on the time dimension:
            - glider_speed_through_water: the speed of the glider through the water
              along its glide path (m s-1)
            - angle_of_attack: the angle between the glider's glide path and its
              heading (degrees)
            - glider_eastward/northward/upward_velocity_through_water: the components of
              the glider's velocity through the water (m s-1)
            - upward_water_velocity: the vertical water velocity (m s-1), the depth rate
              minus the modeled vertical velocity of the glider through the water
        They are NaN where the inputs are missing. They are only valid where the glider
        is in steady flight (see calibration_mask).
    """
    mass = calibration.attrs["mass"]
    rho0 = calibration.attrs["rho0"]
    coefficients = {k: v for k, v in calibration.attrs.items() if k not in ("mass", "rho0")}
    parameters = [
        c
        for c in calibration.columns
        if c not in ("window", "start", "end", "n_points", "rms_dhdt")
    ]
    data, valid = _model_input(ds)
    time = ds["time"].values[valid]

    output = {name: np.full(time.shape, np.nan) for name in ("U", "alpha", "u", "w", "ww")}
    for _, row in calibration.iterrows():
        if not np.isfinite(row[parameters].astype(float)).all():
            continue
        if row["window"] == "deployment":
            in_window = np.ones(time.shape, dtype=bool)
        else:
            in_window = (time >= row["start"]) & (time <= row["end"])
        model = gliderflight.SteadyStateGliderModel(rho0=rho0)
        model.define(mg=mass, **_floats({**coefficients, **row[parameters].to_dict()}))
        result = model.solve({k: v[in_window] for k, v in data.items() if v is not None})
        for name, values in output.items():
            values[in_window] = getattr(result, name)

    heading = np.deg2rad(ds["vehicle_heading"].values[valid])
    variables = {
        "glider_speed_through_water": (
            output["U"],
            "Glider Speed Through the Water",
            "Speed of the glider through the water along its glide path",
        ),
        "angle_of_attack": (
            np.rad2deg(output["alpha"]),
            "Angle of Attack",
            "Angle between the glider's glide path and its heading",
        ),
        "glider_eastward_velocity_through_water": (
            output["u"] * np.sin(heading),
            "Glider Eastward Velocity Through the Water",
            "Eastward component of the glider's horizontal velocity through the water",
        ),
        "glider_northward_velocity_through_water": (
            output["u"] * np.cos(heading),
            "Glider Northward Velocity Through the Water",
            "Northward component of the glider's horizontal velocity through the water",
        ),
        "glider_upward_velocity_through_water": (
            output["w"],
            "Glider Upward Velocity Through the Water",
            "Vertical velocity of the glider through the water, positive up",
        ),
        "upward_water_velocity": (
            output["ww"],
            "Upward Water Velocity",
            (
                "Vertical water velocity, the depth rate minus the modeled vertical velocity of "
                "the glider through the water, positive up"
            ),
        ),
    }
    model_comment = (
        f"Calculated with the steady-state gliderflight model (version "
        f"{gliderflight.__version__}) with a glider mass of {mass} kg. Only valid where the "
        "glider is in steady flight."
    )
    for name, (values, long_name, comment) in variables.items():
        full = np.full(ds.sizes["time"], np.nan)
        full[valid] = values
        units = "degrees" if name == "angle_of_attack" else "m s-1"
        ds[name] = (
            "time",
            full,
            {"long_name": long_name, "units": units, "comment": f"{comment}. {model_comment}"},
        )
    return ds


def flight_model_skill(ds: xr.Dataset, steady: np.ndarray | None = None) -> pd.DataFrame:
    """Compare the glider velocity from the flight model with that measured by the DVL

    Parameters
    ----------
    ds: xarray.Dataset
        The glider DVL dataset with the flight model (solve_flight_model) and DVL
        (dvl_glider_velocity) glider velocities
    steady: numpy.ndarray[bool], Optional
        The measurements to compare. Default is calibration_mask(ds).

    Returns
    -------
    skill: pandas.DataFrame
        For the eastward, northward, upward, and horizontal forward velocities, the
        number of measurements compared, the mean of each, the bias (model minus DVL),
        the rms difference, and the correlation
    """
    if steady is None:
        steady = calibration_mask(ds)
    heading = np.deg2rad(ds["vehicle_heading"].values)
    model_forward = ds["glider_eastward_velocity_through_water"].values * np.sin(heading) + ds[
        "glider_northward_velocity_through_water"
    ].values * np.cos(heading)
    pairs = {
        "eastward": (
            ds["glider_eastward_velocity_through_water"].values,
            ds["dvl_glider_eastward_velocity"].values,
        ),
        "northward": (
            ds["glider_northward_velocity_through_water"].values,
            ds["dvl_glider_northward_velocity"].values,
        ),
        "upward": (
            ds["glider_upward_velocity_through_water"].values,
            ds["dvl_glider_upward_velocity"].values,
        ),
        "forward": (model_forward, ds["dvl_glider_forward_speed"].values),
    }
    rows = []
    for name, (model, dvl) in pairs.items():
        valid = steady & np.isfinite(model) & np.isfinite(dvl)
        m, d = model[valid], dvl[valid]
        row = {"component": name, "n": int(valid.sum())}
        if row["n"] > 1:
            row.update(
                model_mean=m.mean(),
                dvl_mean=d.mean(),
                bias=(m - d).mean(),
                rms_difference=np.sqrt(((m - d) ** 2).mean()),
                correlation=np.corrcoef(m, d)[0, 1],
            )
        rows.append(row)
    return pd.DataFrame(rows).set_index("component")


def _model_input(ds, u_relative=False, pressure="ctd_pressure", density="ctd_density"):
    """The flight model input data (in the gliderflight units) where it is all valid"""
    data = {
        "time": ds["time"].values.astype("datetime64[ns]").astype("int64") / 1e9,
        "pressure": ds[pressure].values / 10,  # dbar to bar
        "pitch": np.deg2rad(ds["vehicle_pitch"].values),
        "buoyancy_change": ds["buoyancy_change"].values.astype(float),
        "density": ds[density].values.astype(float),
    }
    valid = np.logical_and.reduce([np.isfinite(v) for v in data.values()])
    data = {k: v[valid] for k, v in data.items()}
    data["u_relative"] = ds["dvl_glider_forward_speed"].values[valid] if u_relative else None
    return data, valid


def _floats(values):
    """Convert numpy scalars to floats, which gliderflight requires for constant parameters"""
    return {k: float(v) for k, v in values.items() if v is not None}
