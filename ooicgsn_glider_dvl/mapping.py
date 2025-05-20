#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author Andrew Reed
@brief Provides methods for generating maps of glider tracks
"""
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.io.shapereader as shpreader
import cmocean

def map_glider_tracks(bathymetry, glider, **kwargs):
    """Plot the glider track over bathymetry. Optionally include glider waypoints and mooring locations"""

    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    
    # If passed, set map extent
    if 'extant' in kwargs:
        # Set map extent
        extent = kwargs.get('extent')
        ax.set_extent(extent, crs=ccrs.PlateCarree())
    
    # Add in coastlines, borders, and gridlines
    ax.coastlines(resolution="10m")
    ax.add_feature(cfeature.BORDERS)
    ax.gridlines(draw_labels=True)

    
    
    

    return


def get_bathymetry(array):
    return


def get_mooring_locations(array):
    return


def get_elevation_lim(bathymetry):
    """Get the """
    max_depth = bathymetry.min()
    if np.abs(max_depth) < 1000:
        ticks = [
        