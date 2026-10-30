"""
objective_mapping.py

Objective mapping (optimal interpolation) routines for glider data.

This module provides functions for creating objectively mapped cross-sections
from Slocum glider data using optimal interpolation techniques.

Author: [Your Name]
Date: December 2025
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
from scipy.linalg import solve
from scipy.spatial.distance import cdist
import xarray as xr

__version__ = "1.1.0"
__all__ = [
    "MEASUREMENT_ERRORS",
    "covariance_function",
    "estimate_field_variance",
    "km_to_lon",
    "lon_to_km",
    "objective_map_glider",
    "objective_map_section",
    "plot_mapped_section",
]

# Measurement uncertainties for common glider sensors
MEASUREMENT_ERRORS = {
    "ctd_temperature": 0.002,  # °C
    "ctd_salinity": 0.003,  # PSU (converted from conductivity)
    "ctd_conductivity": 0.0003,  # S/m
    "ctd_pressure": 0.1,  # % (will be scaled by actual pressure)
    "ctd_density": 0.01,  # kg/m³
    "sci_oxy4_oxygen": 2.0,  # umol/kg
    "sci_flbbcd_chlor_units": 0.02,  # ug/L (typical)
    "sci_flbbcd_cdom_units": 0.2,  # ppb (typical)
    "sci_flbbcd_bb_units": 0.0002,  # m^-1 sr^-1 (typical)
}


def lon_to_km(lon1, lon2, lat_mean):
    """
    Convert longitude difference to kilometers using local approximation.

    Parameters
    ----------
    lon1 : float or array_like
        First longitude(s) in degrees
    lon2 : float or array_like
        Second longitude(s) in degrees
    lat_mean : float
        Mean latitude for approximation in degrees

    Returns
    -------
    float or ndarray
        Distance in kilometers

    Notes
    -----
    Uses the approximation:
        km = Δlon * cos(lat) * 111.32 km/degree

    where 111.32 km is approximately the length of 1 degree at the equator.
    """
    R_EARTH = 6371.0  # Earth radius in km
    dlon = np.abs(lon2 - lon1)
    # Convert to radians
    lat_rad = np.radians(lat_mean)
    # Distance per degree at this latitude
    km_per_deg = (np.pi / 180.0) * R_EARTH * np.cos(lat_rad)
    return dlon * km_per_deg


def km_to_lon(km, lat_mean):
    """
    Convert kilometers to longitude difference at given latitude.

    Parameters
    ----------
    km : float or array_like
        Distance in kilometers
    lat_mean : float
        Mean latitude for approximation in degrees

    Returns
    -------
    float or ndarray
        Longitude difference in degrees

    Notes
    -----
    Inverse of lon_to_km function.
    """
    R_EARTH = 6371.0  # Earth radius in km
    lat_rad = np.radians(lat_mean)
    km_per_deg = (np.pi / 180.0) * R_EARTH * np.cos(lat_rad)
    return km / km_per_deg


def haversine_distance(lon1, lat1, lon2, lat2):
    """
    Calculate great circle distance between points using Haversine formula.

    Parameters
    ----------
    lon1, lat1 : float or array_like
        Longitude and latitude of first point(s) in degrees
    lon2, lat2 : float or array_like
        Longitude and latitude of second point(s) in degrees

    Returns
    -------
    float or ndarray
        Distance in kilometers

    Notes
    -----
    More accurate than simple approximation for larger distances.
    """
    R_EARTH = 6371.0  # Earth radius in km

    # Convert to radians
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])

    # Haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))

    return R_EARTH * c


def covariance_function(r, length_scale, variance, cov_type="gaussian"):
    """
    Calculate covariance as a function of distance.

    Parameters
    ----------
    r : array_like
        Distance(s) between points
    length_scale : float
        Correlation length scale
    variance : float
        Field variance (maximum covariance at r=0)
    cov_type : str, optional
        Type of covariance function ('gaussian' or 'exponential')
        Default is 'gaussian'

    Returns
    -------
    ndarray
        Covariance values at distances r

    Raises
    ------
    ValueError
        If cov_type is not 'gaussian' or 'exponential'

    Notes
    -----
    Gaussian: C(r) = σ² * exp(-r²/(2L²))
    Exponential: C(r) = σ² * exp(-r/L)

    where σ² is variance and L is length_scale
    """
    if cov_type == "gaussian":
        return variance * np.exp(-(r**2) / (2 * length_scale**2))
    elif cov_type == "exponential":
        return variance * np.exp(-r / length_scale)
    else:
        raise ValueError(
            f"Unknown covariance type: {cov_type}. Must be 'gaussian' or 'exponential'"
        )


def calculate_separation_matrix(coords1, coords2, length_scales, lat_mean=None, use_km=False):
    """
    Calculate normalized separation matrix between coordinate sets.

    Parameters
    ----------
    coords1 : ndarray, shape (n1, 2)
        First set of coordinates [horizontal, vertical]
    coords2 : ndarray, shape (n2, 2)
        Second set of coordinates [horizontal, vertical]
    length_scales : tuple of float
        (horizontal_length_scale, vertical_length_scale)
    lat_mean : float, optional
        Mean latitude in degrees (required if use_km=True for longitude data)
    use_km : bool, optional
        If True, convert longitude differences to kilometers before computing distances
        Default is False

    Returns
    -------
    ndarray, shape (n1, n2)
        Euclidean separation distances in normalized coordinate space

    Notes
    -----
    Coordinates are normalized by their respective length scales before
    computing distances, allowing for anisotropic correlation structures.

    When use_km=True, horizontal coordinates (assumed to be longitude in degrees)
    are converted to kilometers using a local approximation at lat_mean before
    normalization.
    """
    # Normalize coordinates by length scales
    coords1_norm = coords1.copy()
    coords2_norm = coords2.copy()

    if use_km and lat_mean is not None:
        # Convert longitude to km for coords1
        lon_ref = coords1[0, 0]  # Reference longitude
        km1 = lon_to_km(coords1[:, 0], lon_ref, lat_mean)
        km2 = lon_to_km(coords2[:, 0], lon_ref, lat_mean)

        coords1_norm[:, 0] = km1 / length_scales[0]
        coords2_norm[:, 0] = km2 / length_scales[0]
    else:
        coords1_norm[:, 0] = coords1[:, 0] / length_scales[0]
        coords2_norm[:, 0] = coords2[:, 0] / length_scales[0]

    coords1_norm[:, 1] = coords1[:, 1] / length_scales[1]
    coords2_norm[:, 1] = coords2[:, 1] / length_scales[1]

    # Calculate Euclidean distance in normalized space
    dist_matrix = cdist(coords1_norm, coords2_norm, metric="euclidean")

    return dist_matrix


def estimate_field_variance(obs_values, method="detrended"):
    """
    Estimate background field variance from observations.

    Parameters
    ----------
    obs_values : array_like
        Observed values
    method : str, optional
        Method for variance estimation:
        - 'simple': use full variance
        - 'detrended': remove linear trend first (default, more conservative)

    Returns
    -------
    float
        Estimated field variance

    Notes
    -----
    The 'detrended' method removes large-scale trends that may inflate
    variance estimates, providing a more conservative estimate of the
    background variability.
    """
    # Remove NaNs
    values = obs_values[~np.isnan(obs_values)]

    if len(values) < 2:
        return 1.0  # Default if insufficient data

    if method == "simple":
        return np.var(values)
    elif method == "detrended":
        # Fit and remove linear trend
        x = np.arange(len(values))
        coeffs = np.polyfit(x, values, 1)
        trend = np.polyval(coeffs, x)
        detrended = values - trend
        return np.var(detrended)
    else:
        return np.var(values)


def objective_map_glider(
    obs_coords,
    obs_values,
    grid_coords,
    length_scales,
    variable_name="ctd_salinity",
    noise_variance=None,
    field_variance=None,
    covariance_type="gaussian",
    return_errors=True,
    lat_mean=None,
    use_km=False,
):
    """
    Objective mapping (optimal interpolation) for glider data.

    Performs optimal interpolation using specified covariance structure
    to map irregularly spaced observations onto a regular grid.

    Parameters
    ----------
    obs_coords : ndarray, shape (n_obs, 2)
        Observation coordinates [horizontal, vertical]
        horizontal: longitude (degrees) or time (seconds)
        vertical: depth/pressure (meters or dbar)
    obs_values : ndarray, shape (n_obs,)
        Observed values at observation points
    grid_coords : ndarray, shape (n_grid, 2)
        Grid points to map onto [horizontal, vertical]
    length_scales : tuple of float
        (horizontal_length_scale, vertical_length_scale)
        For longitude data with use_km=True: horizontal scale in km
        For longitude data with use_km=False: horizontal scale in degrees
        For time data: horizontal scale in seconds
        Vertical scale always in meters or dbar
    variable_name : str, optional
        Variable being mapped (for error lookup), default 'ctd_salinity'
    noise_variance : float, optional
        Observation error variance. If None, uses MEASUREMENT_ERRORS
    field_variance : float, optional
        Background field variance. If None, estimated from data
    covariance_type : str, optional
        'gaussian' or 'exponential', default 'gaussian'
    return_errors : bool, optional
        If True, return mapping errors, default True
    lat_mean : float, optional
        Mean latitude in degrees (required if use_km=True for longitude data)
    use_km : bool, optional
        If True, convert longitude to km before computing distances
        Default is False

    Returns
    -------
    mapped_values : ndarray, shape (n_grid,)
        Objectively mapped values at grid points
    mapping_errors : ndarray, shape (n_grid,), optional
        Mapping error standard deviation (only if return_errors=True)

    Notes
    -----
    The objective mapping solution is:
        x_mapped = C_md @ (C_dd + N)^(-1) @ x_obs

    where:
        C_md: model-data covariance
        C_dd: data-data covariance
        N: observation error covariance (diagonal)
        x_obs: observations

    The mapping error variance is:
        σ²_error = σ²_prior - C_md @ (C_dd + N)^(-1) @ C_md^T

    References
    ----------
    Bretherton, F. P., R. E. Davis, and C. B. Fandry, 1976: A technique for
    objective analysis and design of oceanographic experiments applied to
    MODE-73. Deep-Sea Res., 23, 559-582.
    """

    # Remove NaN observations
    valid_mask = ~np.isnan(obs_values)
    obs_coords = obs_coords[valid_mask]
    obs_values = obs_values[valid_mask]

    n_obs = len(obs_values)
    n_grid = len(grid_coords)

    if n_obs == 0:
        # No valid observations
        if return_errors:
            return np.full(n_grid, np.nan), np.full(n_grid, np.nan)
        else:
            return np.full(n_grid, np.nan)

    # Estimate noise variance from measurement error
    if noise_variance is None:
        if variable_name in MEASUREMENT_ERRORS:
            noise_std = MEASUREMENT_ERRORS[variable_name]
            # For pressure, scale by actual pressure value
            if "pressure" in variable_name.lower():
                # Convert percentage to absolute
                mean_pressure = np.nanmean(obs_coords[:, 1])
                noise_std = (noise_std / 100.0) * mean_pressure
            noise_variance = noise_std**2
        else:
            # Default to 1% of field variance if variable not found
            noise_variance = 0.01 * np.var(obs_values)

    # Estimate field variance
    if field_variance is None:
        field_variance = estimate_field_variance(obs_values, method="detrended")

    # Use average length scale for covariance calculation
    L_avg = np.mean(length_scales)

    # Calculate observation-observation covariance matrix (C_dd)
    r_dd = calculate_separation_matrix(
        obs_coords, obs_coords, length_scales, lat_mean=lat_mean, use_km=use_km
    )
    C_dd = covariance_function(r_dd, L_avg, field_variance, covariance_type)

    # Add noise to diagonal (observation error)
    C_dd_noisy = C_dd + noise_variance * np.eye(n_obs)

    # Calculate grid-observation covariance matrix (C_md)
    r_md = calculate_separation_matrix(
        grid_coords, obs_coords, length_scales, lat_mean=lat_mean, use_km=use_km
    )
    C_md = covariance_function(r_md, L_avg, field_variance, covariance_type)

    # Solve for weights: C_dd_noisy @ weights = C_md.T
    try:
        weights = solve(C_dd_noisy, C_md.T, assume_a="pos")
    except np.linalg.LinAlgError:
        # If matrix is singular, add small regularization
        print("Warning: Singular covariance matrix, adding regularization")
        reg = 1e-6 * np.mean(np.diag(C_dd_noisy))
        C_dd_noisy += reg * np.eye(n_obs)
        weights = solve(C_dd_noisy, C_md.T, assume_a="pos")

    # Calculate mapped values
    mapped_values = weights.T @ obs_values

    if return_errors:
        # Calculate mapping error variance
        error_variance = field_variance - np.sum(C_md * weights.T, axis=1)

        # Ensure non-negative (numerical errors can make it slightly negative)
        error_variance = np.maximum(error_variance, 0)
        mapping_errors = np.sqrt(error_variance)

        return mapped_values, mapping_errors
    else:
        return mapped_values


def objective_map_section(
    ds,
    variable,
    grid_type="longitude",
    horizontal_scale=10.0,
    vertical_scale=10.0,
    n_horizontal=100,
    depth_resolution=1.0,
    covariance_type="gaussian",
    subset_dict=None,
    use_km=True,
):
    """
    Create objectively mapped cross-section from glider dataset.

    High-level function that handles coordinate extraction, grid creation,
    and xarray Dataset formatting for mapped sections.

    Parameters
    ----------
    ds : xarray.Dataset
        Glider dataset with time dimension
    variable : str
        Variable name to map (e.g., 'ctd_salinity', 'ctd_temperature')
    grid_type : str, optional
        'longitude' for lon-depth section or 'time' for time-depth section
        Default is 'longitude'
    horizontal_scale : float, optional
        Horizontal correlation length scale
        For longitude with use_km=True: kilometers (default)
        For longitude with use_km=False: degrees
        For time: hours
        Default is 10.0
    vertical_scale : float, optional
        Vertical correlation length scale (meters or dbar)
        Default is 10.0
    n_horizontal : int, optional
        Number of horizontal grid points, default 100
    depth_resolution : float, optional
        Vertical grid resolution (meters or dbar), default 1.0
    covariance_type : str, optional
        'gaussian' or 'exponential', default 'gaussian'
    subset_dict : dict, optional
        Dictionary for subsetting dataset, e.g., {'transect_id': 1}
    use_km : bool, optional
        If True (default), interpret horizontal_scale as kilometers for longitude data
        If False, interpret as degrees

    Returns
    -------
    xarray.Dataset
        Dataset with mapped field and errors on regular grid
        Contains variables:
        - {variable}_mapped: objectively mapped field
        - {variable}_error: posterior error standard deviation

    Examples
    --------
    >>> import xarray as xr
    >>> ds = xr.open_dataset('glider_data.nc')
    >>>
    >>> # Map salinity on longitude-depth section (horizontal scale in km)
    >>> ds_sal = objective_map_section(
    ...     ds,
    ...     variable='ctd_salinity',
    ...     grid_type='longitude',
    ...     horizontal_scale=10.0,  # 10 km
    ...     vertical_scale=20.0,
    ...     use_km=True
    ... )
    >>>
    >>> # Map temperature on time-depth section for specific transect
    >>> ds_temp = objective_map_section(
    ...     ds,
    ...     variable='ctd_temperature',
    ...     grid_type='time',
    ...     horizontal_scale=2.0,  # 2 hours
    ...     vertical_scale=15.0,
    ...     subset_dict={'transect_id': 1}
    ... )
    """

    # Subset data if requested
    if subset_dict is not None:
        for key, value in subset_dict.items():
            ds = ds.where(ds[key] == value, drop=True)

    # Get coordinates
    if grid_type == "longitude":
        horizontal_coord = "lon"
        h_values = ds[horizontal_coord].values
        h_label = "Longitude"
        h_units = "degrees_east"

        # Calculate mean latitude for km conversion
        if "lat" in ds.coords:
            lat_mean = float(np.nanmean(ds.coords["lat"].values))
        elif "lat" in ds.data_vars:
            lat_mean = float(np.nanmean(ds["lat"].values))
        else:
            raise ValueError("Could not find 'lat' in dataset")

    elif grid_type == "time":
        horizontal_coord = "time"
        h_values = ds[horizontal_coord].values
        # Convert time to seconds since start
        time_start = h_values.min()
        h_values = (h_values - time_start) / np.timedelta64(1, "s")
        horizontal_scale = horizontal_scale * 3600  # Convert hours to seconds
        h_label = "Time"
        h_units = "datetime64[ns]"
        lat_mean = None
        use_km = False  # Not applicable for time
    else:
        raise ValueError(f"Unknown grid_type: {grid_type}. Must be 'longitude' or 'time'")

    # Get vertical coordinate (try both pressure and depth)
    if "ctd_pressure" in ds.variables:
        vertical_coord = "ctd_pressure"
        v_values = ds[vertical_coord].values
        v_label = "Pressure"
        v_units = "dbar"
    elif "depth" in ds.coords:
        vertical_coord = "depth"
        v_values = ds.coords[vertical_coord].values
        v_label = "Depth"
        v_units = "meters"
    else:
        raise ValueError("Could not find 'ctd_pressure' or 'depth' in dataset")

    # Get variable values
    if variable not in ds.variables:
        raise ValueError(f"Variable '{variable}' not found in dataset")

    var_values = ds[variable].values

    # Create observation coordinate array
    valid_mask = ~np.isnan(var_values) & ~np.isnan(h_values) & ~np.isnan(v_values)
    obs_coords = np.column_stack([h_values[valid_mask], v_values[valid_mask]])
    obs_values = var_values[valid_mask]

    if len(obs_values) == 0:
        raise ValueError("No valid observations found after filtering NaNs")

    # Create regular grid
    h_min, h_max = np.nanmin(h_values), np.nanmax(h_values)
    v_min, v_max = 0, np.ceil(np.nanmax(v_values) / 10) * 10

    h_grid = np.linspace(h_min, h_max, n_horizontal)
    v_grid = np.arange(v_min, v_max + depth_resolution, depth_resolution)

    H_GRID, V_GRID = np.meshgrid(h_grid, v_grid)
    grid_coords = np.column_stack([H_GRID.ravel(), V_GRID.ravel()])

    # Perform objective mapping
    print(f"Mapping {variable} using {len(obs_values)} observations...")
    if grid_type == "longitude" and use_km:
        print(
            f"Length scales: horizontal={horizontal_scale:.2f} km, vertical={vertical_scale:.2f} m"
        )
        print(f"Mean latitude: {lat_mean:.4f}°")
    elif grid_type == "longitude":
        print(
            f"Length scales: horizontal={horizontal_scale:.4f} degrees, vertical={vertical_scale:.2f} m"
        )
    else:
        print(
            f"Length scales: horizontal={horizontal_scale / 3600:.2f} hours, vertical={vertical_scale:.2f} m"
        )
    print(f"Grid size: {len(v_grid)} x {len(h_grid)}")

    mapped_values, mapping_errors = objective_map_glider(
        obs_coords,
        obs_values,
        grid_coords,
        length_scales=(horizontal_scale, vertical_scale),
        variable_name=variable,
        covariance_type=covariance_type,
        return_errors=True,
        lat_mean=lat_mean,
        use_km=use_km,
    )

    # Reshape to grid
    mapped_field = mapped_values.reshape(H_GRID.shape)
    error_field = mapping_errors.reshape(H_GRID.shape)

    # Create output dataset
    if grid_type == "time":
        # Convert back to datetime
        h_grid_datetime = time_start + h_grid * np.timedelta64(1, "s")
        h_coord = h_grid_datetime
    else:
        h_coord = h_grid

    ds_out = xr.Dataset(
        {
            f"{variable}_mapped": ([v_label.lower(), horizontal_coord], mapped_field),
            f"{variable}_error": ([v_label.lower(), horizontal_coord], error_field),
        },
        coords={horizontal_coord: h_coord, v_label.lower(): v_grid},
    )

    # Add attributes
    h_scale_attr = (
        f"{horizontal_scale:.2f} km"
        if (grid_type == "longitude" and use_km)
        else f"{horizontal_scale:.4f}"
    )

    ds_out[f"{variable}_mapped"].attrs = {
        "long_name": f"Objectively Mapped {variable}",
        "units": ds[variable].attrs.get("units", ""),
        "horizontal_scale": h_scale_attr,
        "vertical_scale": vertical_scale,
        "covariance_type": covariance_type,
        "n_observations": len(obs_values),
    }

    ds_out[f"{variable}_error"].attrs = {
        "long_name": f"Mapping Error for {variable}",
        "units": ds[variable].attrs.get("units", ""),
        "description": "Posterior error standard deviation",
    }

    ds_out.coords[horizontal_coord].attrs = {"long_name": h_label, "units": h_units}

    ds_out.coords[v_label.lower()].attrs = {"long_name": v_label, "units": v_units}

    print("Mapping complete!")

    return ds_out


def plot_mapped_section(
    ds_mapped,
    variable,
    grid_type="longitude",
    plot_errors=True,
    mask_bottom=True,
    bottom_data=None,
    figsize=None,
    **kwargs,
):
    """
    Plot objectively mapped section with optional error estimates.

    Parameters
    ----------
    ds_mapped : xarray.Dataset
        Output from objective_map_section
    variable : str
        Variable name that was mapped
    grid_type : str, optional
        'longitude' or 'time', default 'longitude'
    plot_errors : bool, optional
        If True, create subplot with mapping errors, default True
    mask_bottom : bool, optional
        If True, mask values below bottom, default True
    bottom_data : tuple of array_like, optional
        (horizontal_coords, bottom_depths) for masking
    figsize : tuple, optional
        Figure size (width, height) in inches
    **kwargs : dict
        Additional arguments for pcolormesh (cmap, vmin, vmax, etc.)

    Returns
    -------
    matplotlib.figure.Figure
        Figure object containing the plot(s)

    Examples
    --------
    >>> fig = plot_mapped_section(
    ...     ds_sal,
    ...     'ctd_salinity',
    ...     cmap='viridis',
    ...     vmin=32, vmax=36
    ... )
    >>> plt.show()
    """

    if grid_type == "longitude":
        h_coord = "lon"
        h_label = "Longitude (°E)"
    else:
        h_coord = "time"
        h_label = "Time"

    # Determine vertical coordinate
    if "pressure" in ds_mapped.coords:
        v_coord = "pressure"
        v_label = "Pressure (dbar)"
    else:
        v_coord = "depth"
        v_label = "Depth (m)"

    # Get data
    mapped_field = ds_mapped[f"{variable}_mapped"]
    error_field = ds_mapped[f"{variable}_error"]

    # Mask bottom if requested
    if mask_bottom and bottom_data is not None:
        h_bottom, v_bottom = bottom_data

        f_bottom = interp1d(
            h_bottom, v_bottom, kind="linear", fill_value="extrapolate", bounds_error=False
        )

        h_grid = ds_mapped[h_coord].values
        if grid_type == "time":
            # Convert datetime to seconds for interpolation
            h_grid_numeric = (h_grid - h_grid[0]) / np.timedelta64(1, "s")
            bottom_interp = f_bottom(h_grid_numeric)
        else:
            bottom_interp = f_bottom(h_grid)

        # Create mask
        v_grid = ds_mapped[v_coord].values
        V_GRID = np.tile(v_grid[:, np.newaxis], (1, len(h_grid)))
        BOTTOM_GRID = np.tile(bottom_interp[np.newaxis, :], (len(v_grid), 1))

        mapped_field = mapped_field.where(V_GRID <= BOTTOM_GRID)

    # Create figure
    if figsize is None:
        figsize = (14, 6) if plot_errors else (10, 6)

    if plot_errors:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    else:
        fig, ax1 = plt.subplots(1, 1, figsize=figsize)

    # Plot mapped field
    mapped_field.plot(ax=ax1, x=h_coord, y=v_coord, yincrease=False, add_colorbar=True, **kwargs)
    ax1.set_xlabel(h_label)
    ax1.set_ylabel(v_label)
    ax1.set_title(f"Objectively Mapped {variable}")

    if mask_bottom and bottom_data is not None:
        ax1.plot(h_grid, bottom_interp, "k-", linewidth=2, label="Bottom")
        ax1.legend()

    # Plot errors
    if plot_errors:
        # Remove incompatible kwargs for error plot
        error_kwargs = {
            k: v for k, v in kwargs.items() if k not in ["vmin", "vmax", "levels", "extend"]
        }
        error_kwargs["cmap"] = error_kwargs.get("cmap", "Reds")

        error_field.plot(
            ax=ax2, x=h_coord, y=v_coord, yincrease=False, add_colorbar=True, **error_kwargs
        )
        ax2.set_xlabel(h_label)
        ax2.set_ylabel(v_label)
        ax2.set_title("Mapping Error")

        if mask_bottom and bottom_data is not None:
            ax2.plot(h_grid, bottom_interp, "k-", linewidth=2)

    plt.tight_layout()
    return fig


if __name__ == "__main__":
    print("glider_objmap module loaded successfully")
    print(f"Version: {__version__}")
    print("\nAvailable functions:")
    for func in __all__:
        print(f"  - {func}")
    print("\nFor usage examples, see function docstrings:")
    print("  help(objective_map_section)")
    print("  help(plot_mapped_section)")
