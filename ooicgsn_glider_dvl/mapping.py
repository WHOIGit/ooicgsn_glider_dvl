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
    waypoints: bool | pd.DataFrame = True,
    waypoint_table: bool = True,
    levels: list | None = None,
    title: str | None = None,
    pad: float = 0.05,
):
    """Plot the glider track over bathymetry with the glider waypoints and mooring locations.

    The track is drawn as a black line, the waypoints as yellow stars, and the moorings as
    red circles, over the bathymetry shaded by elevation with a horizontal colorbar. A
    table of the waypoint coordinates can be added to the top right corner of the map.

    Args:
        glider: Glider dataset with glider_precise_lon/lat and (optionally) waypoint_lon/lat
        bathymetry: Elevation (m, negative below sea level) on latitude/longitude, e.g. from
            get_bathymetry. Land (elevation >= 0) is not shaded.
        moorings: Table with lat and lon columns, e.g. from get_mooring_locations
        extent: Map extent as [lon_min, lon_max, lat_min, lat_max]. Defaults to the glider
            track, waypoints, and moorings padded by pad degrees.
        waypoints: True plots the glider's waypoints (from get_waypoints), False plots none,
            or pass a table with lat and lon columns to plot those waypoints instead
        waypoint_table: Add a table of the waypoint coordinates to the map
        levels: Elevation contour levels (m) for the bathymetry. Defaults to get_elevation_lim.
        title: Figure title
        pad: Padding (degrees) around the default extent

    Returns:
        fig, ax: The matplotlib figure and cartopy GeoAxes
    """
    # Drop the missing positions so the track is drawn as a continuous line
    track_lon = np.ravel(glider["glider_precise_lon"].values)
    track_lat = np.ravel(glider["glider_precise_lat"].values)
    valid = np.isfinite(track_lon) & np.isfinite(track_lat)
    track_lon, track_lat = track_lon[valid], track_lat[valid]

    # Get the waypoints to plot
    if isinstance(waypoints, pd.DataFrame):
        wpts = waypoints[["lat", "lon"]].reset_index(drop=True)
    elif waypoints:
        wpts = get_waypoints(glider)
    else:
        wpts = pd.DataFrame(columns=["lat", "lon"])

    if extent is None:
        lons = [track_lon.min(), track_lon.max(), *wpts["lon"]]
        lats = [track_lat.min(), track_lat.max(), *wpts["lat"]]
        if moorings is not None:
            lons += [moorings["lon"].min(), moorings["lon"].max()]
            lats += [moorings["lat"].min(), moorings["lat"].max()]
        extent = [
            float(np.nanmin(lons)) - pad,
            float(np.nanmax(lons)) + pad,
            float(np.nanmin(lats)) - pad,
            float(np.nanmax(lats)) + pad,
        ]

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    ax.set_extent(extent, crs=ccrs.PlateCarree())

    # Shade the bathymetry by elevation below sea level
    if bathymetry is not None:
        elevation = bathymetry.where(bathymetry < 0)
        if levels is None:
            levels = get_elevation_lim(bathymetry)
        levels = np.asarray(levels)
        extend = "min" if float(bathymetry.min()) < levels.min() else "neither"
        cf = ax.contourf(
            elevation["longitude"],
            elevation["latitude"],
            elevation,
            levels=levels,
            cmap=cmocean.cm.deep_r,
            extend=extend,
            transform=ccrs.PlateCarree(),
        )
        fig.colorbar(
            cf,
            ax=ax,
            orientation="horizontal",
            label="Water depth (m)",
            ticks=_colorbar_ticks(levels),
            pad=0.05,
        )

    # Add in land, coastlines, borders, and gridlines
    ax.add_feature(cfeature.LAND.with_scale("10m"), facecolor="lightgray", zorder=1)
    ax.coastlines(resolution="10m", zorder=2)
    ax.add_feature(cfeature.BORDERS, zorder=2)
    ax.gridlines(draw_labels=True)

    # Plot the glider track, mooring locations, and waypoints
    ax.plot(track_lon, track_lat, color="black", transform=ccrs.PlateCarree(), zorder=3)

    if moorings is not None:
        ax.plot(
            moorings["lon"],
            moorings["lat"],
            marker="o",
            linestyle="",
            markeredgecolor="black",
            color="tab:red",
            transform=ccrs.PlateCarree(),
            zorder=4,
        )

    if not wpts.empty:
        ax.plot(
            wpts["lon"],
            wpts["lat"],
            marker="*",
            linestyle="",
            markersize=20,
            markeredgecolor="black",
            color="yellow",
            transform=ccrs.PlateCarree(),
            zorder=5,
        )
        if waypoint_table:
            add_waypoint_table(ax, wpts)

    if title is not None:
        ax.set_title(title)

    return fig, ax


def get_waypoints(glider: xr.Dataset) -> pd.DataFrame:
    """Get the glider's waypoints, in the order they were first set.

    Args:
        glider: Glider dataset with waypoint_lat and waypoint_lon

    Returns:
        Table of the unique waypoint lat and lon, dropping missing and (0, 0) entries
    """
    if "waypoint_lat" not in glider or "waypoint_lon" not in glider:
        return pd.DataFrame(columns=["lat", "lon"])

    wpts = pd.DataFrame({
        "lat": np.ravel(glider["waypoint_lat"].values),
        "lon": np.ravel(glider["waypoint_lon"].values),
    }).dropna()
    wpts = wpts[(wpts["lat"] != 0) | (wpts["lon"] != 0)]
    return wpts.drop_duplicates().reset_index(drop=True)


def add_waypoint_table(ax, waypoints: pd.DataFrame, fontsize: int = 9):
    """Add a table of the waypoint coordinates to the top right corner of a map.

    Args:
        ax: The map axes
        waypoints: Table with lat and lon columns
        fontsize: Font size of the table text

    Returns:
        The matplotlib table
    """
    # Size the table to the number of waypoints, in fractions of the map axes
    row_height = 0.0625
    height = row_height * (len(waypoints) + 1)
    table_ax = ax.inset_axes([0.74, 0.93 - height, 0.26, height])
    table_ax.axis("off")

    cell_text = [
        [f"{lat:.6f}", f"{lon:.6f}"] for lat, lon in zip(waypoints["lat"], waypoints["lon"])
    ]
    table = table_ax.table(
        cellText=cell_text,
        colLabels=["Latitude", "Longitude"],
        cellLoc="center",
        loc="center",
        bbox=[0, 0, 1, 1],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(fontsize)

    # Style the header row, then shade alternating data rows
    for (row, _), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor("#40466e")
            cell.set_text_props(weight="bold", color="white")
        else:
            cell.set_facecolor("#f0f0f0" if row % 2 == 0 else "white")

    table_ax.text(
        0.5,
        1.02,
        "Waypoint Coordinates",
        transform=table_ax.transAxes,
        fontsize=fontsize + 1,
        fontweight="bold",
        ha="center",
        va="bottom",
    )
    return table


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
    # Pad the region by two grid cells so the shading reaches the edges of the map
    grid_pad = 2 / 60
    lon_min, lon_max, lat_min, lat_max = extent
    lon_min, lon_max = lon_min - grid_pad, lon_max + grid_pad
    lat_min, lat_max = lat_min - grid_pad, lat_max + grid_pad

    # Name the saved file by the padded region, so files saved without the padding aren't reused
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


NICE_STEPS = (1, 2, 5, 10, 25, 50, 100, 250, 500, 1000)


def get_elevation_lim(bathymetry: xr.DataArray, n_levels: int = 50) -> np.ndarray:
    """Get the elevation contour levels for shading the bathymetry.

    The contour spacing is the smallest round step that gives at most n_levels levels
    down to the maximum depth, e.g. 50 m for a region 2000 m deep, so that shallow
    shelf regions still show structure.

    Args:
        bathymetry: Elevation (m), negative below sea level
        n_levels: The most contour levels to use

    Returns:
        Elevation levels (m), from below the maximum depth up to 0
    """
    max_depth = max(-float(bathymetry.min()), 1)
    step = next((s for s in NICE_STEPS if max_depth / s <= n_levels), NICE_STEPS[-1])
    bottom = np.ceil(max_depth / step) * step
    return np.arange(-bottom, step / 2, step)


def _colorbar_ticks(levels: np.ndarray, n_ticks: int = 5) -> np.ndarray:
    """Round colorbar ticks every few contour levels, ending at the top level"""
    span = levels.max() - levels.min()
    step = next((s for s in NICE_STEPS if span / s <= n_ticks), NICE_STEPS[-1])
    return np.arange(levels.max(), levels.min() - step / 2, -step)[::-1]
