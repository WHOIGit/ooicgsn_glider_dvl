#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author Andrew Reed
@brief Provides methods for generating maps of glider tracks
"""

from pathlib import Path

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cmocean
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

# NOAA ETOPO1 global relief (1 arc-minute) served via ERDDAP OPeNDAP
BATHYMETRY_URL = "https://coastwatch.pfeg.noaa.gov/erddap/griddap/etopo180"

# Third-party data, including cached bathymetry downloads
DATA_EXTERNAL = Path(__file__).parents[1] / "data" / "external"

# Mooring locations saved by notebooks/1-data-access-and-download.ipynb
MOORING_LOCATIONS = DATA_EXTERNAL / "mooring_locations.csv"


def map_glider_tracks(
    glider: xr.Dataset,
    bathymetry: xr.DataArray | None = None,
    moorings: pd.DataFrame | None = None,
    extent: list | None = None,
    waypoints: bool = True,
    levels: list | None = None,
    title: str | None = None,
):
    """Plot the glider track, optionally over bathymetry with glider waypoints and mooring locations.

    Args:
        glider: Glider dataset with glider_precise_lon/lat and (optionally) waypoint_lon/lat
        bathymetry: Elevation (m, negative below sea level) on latitude/longitude, e.g. from
            get_bathymetry. Shaded as depth if passed.
        moorings: Table with mooring, lat, and lon columns, e.g. from get_mooring_locations
        extent: Map extent as [lon_min, lon_max, lat_min, lat_max]. Defaults to the glider
            track padded by 0.25 degrees.
        waypoints: Plot the glider waypoints, dropping invalid (0.0) entries
        levels: Depth contour levels (m) for the bathymetry. Defaults to get_elevation_lim.
        title: Figure title

    Returns:
        fig, ax: The matplotlib figure and cartopy GeoAxes
    """
    track_lon = glider["glider_precise_lon"]
    track_lat = glider["glider_precise_lat"]

    if extent is None:
        pad = 0.25
        extent = [
            float(track_lon.min()) - pad,
            float(track_lon.max()) + pad,
            float(track_lat.min()) - pad,
            float(track_lat.max()) + pad,
        ]

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    ax.set_extent(extent, crs=ccrs.PlateCarree())

    # Shade the bathymetry as positive depth below sea level
    if bathymetry is not None:
        depth = -bathymetry.where(bathymetry < 0)
        if levels is None:
            levels = get_elevation_lim(bathymetry)
        cf = ax.contourf(
            depth["longitude"],
            depth["latitude"],
            depth,
            levels=levels,
            cmap=cmocean.cm.deep,
            extend="max",
            transform=ccrs.PlateCarree(),
        )
        fig.colorbar(cf, ax=ax, shrink=0.6, label="Depth (m)")

    # Add in land, coastlines, borders, and gridlines
    ax.add_feature(cfeature.LAND.with_scale("10m"), facecolor="lightgray", zorder=1)
    ax.coastlines(resolution="10m", zorder=2)
    ax.add_feature(cfeature.BORDERS, linestyle=":", zorder=2)
    gl = ax.gridlines(draw_labels=True, linestyle="--", alpha=0.5)
    gl.top_labels = False
    gl.right_labels = False

    ax.plot(
        track_lon,
        track_lat,
        color="green",
        marker="o",
        markersize=3,
        label="Glider Track",
        transform=ccrs.PlateCarree(),
        zorder=3,
    )

    if waypoints and "waypoint_lon" in glider and "waypoint_lat" in glider:
        waypoint_lon = glider["waypoint_lon"]
        waypoint_lat = glider["waypoint_lat"]
        valid = (waypoint_lon != 0.0) & (waypoint_lat != 0.0)
        ax.plot(
            waypoint_lon.where(valid, drop=True),
            waypoint_lat.where(valid, drop=True),
            color="red",
            marker="*",
            markersize=12,
            linestyle="None",
            label="Waypoints",
            transform=ccrs.PlateCarree(),
            zorder=4,
        )

    if moorings is not None:
        ax.plot(
            moorings["lon"],
            moorings["lat"],
            color="blue",
            marker="x",
            markersize=8,
            linestyle="None",
            label="Mooring Location",
            transform=ccrs.PlateCarree(),
            zorder=4,
        )
        for _, mooring in moorings.iterrows():
            ax.annotate(
                mooring["mooring"],
                xy=(mooring["lon"], mooring["lat"]),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=8,
                transform=ccrs.PlateCarree(),
                zorder=4,
            )

    if title is not None:
        ax.set_title(title)
    ax.legend(loc="lower right")

    return fig, ax


def get_bathymetry(
    extent: list,
    stride: int = 1,
    cache_dir=DATA_EXTERNAL,
    overwrite: bool = False,
) -> xr.DataArray:
    """Download ETOPO1 bathymetry from the NOAA CoastWatch ERDDAP, caching it locally.

    The download is saved as a netCDF file in cache_dir, named by the extent and stride.
    Later calls for the same region load the saved file instead of contacting ERDDAP.

    Args:
        extent: Region as [lon_min, lon_max, lat_min, lat_max]
        stride: Subsample every nth grid point (1 arc-minute resolution)
        cache_dir: Directory for the saved bathymetry. Pass None to skip caching.
        overwrite: Re-download even if a saved file exists

    Returns:
        Elevation (m), negative below sea level, on latitude/longitude
    """
    lon_min, lon_max, lat_min, lat_max = extent

    cache_file = None
    if cache_dir is not None:
        name = f"etopo1_{lon_min:.3f}_{lon_max:.3f}_{lat_min:.3f}_{lat_max:.3f}_s{stride}.nc"
        cache_file = Path(cache_dir) / name
        if cache_file.exists() and not overwrite:
            return xr.load_dataarray(cache_file)

    with xr.open_dataset(BATHYMETRY_URL) as ds:
        bathymetry = ds["altitude"].sel(
            latitude=slice(lat_min, lat_max),
            longitude=slice(lon_min, lon_max),
        )
        bathymetry = bathymetry[::stride, ::stride].load()

    if cache_file is not None:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        bathymetry.to_netcdf(cache_file)
    return bathymetry


def get_mooring_locations(array: str | None = None, path=MOORING_LOCATIONS) -> pd.DataFrame:
    """Load the OOI mooring locations, optionally subset to an array.

    Args:
        array: Mooring name prefix to select, e.g. "CP0" for the Pioneer-NES moorings
        path: CSV of mooring, lat, lon, and depth generated by notebook 1

    Returns:
        Table of mooring locations
    """
    moorings = pd.read_csv(path)
    if array is not None:
        moorings = moorings[moorings["mooring"].str.startswith(array)]
    return moorings.reset_index(drop=True)


def get_elevation_lim(bathymetry: xr.DataArray) -> np.ndarray:
    """Get the depth contour levels for shading the bathymetry.

    The contour spacing scales with the maximum depth in the region so that shallow
    shelf regions still show structure.

    Args:
        bathymetry: Elevation (m), negative below sea level

    Returns:
        Depth levels (m), from 0 to the maximum depth
    """
    max_depth = np.abs(float(bathymetry.min()))
    if max_depth < 200:
        step = 10
    elif max_depth < 1000:
        step = 50
    elif max_depth < 3000:
        step = 250
    else:
        step = 500
    return np.arange(0, np.ceil(max_depth / step) * step + step, step)
