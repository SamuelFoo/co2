from pathlib import Path
from typing import List

import contextily as cx
import geopandas as gpd
import matplotlib.pyplot as plt
from geopandas.geodataframe import GeoDataFrame
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib_scalebar.scalebar import ScaleBar


def get_co2_axes(ax: Axes) -> Axes:
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("CO2 (ppm)")
    return ax


def plot_co2(ts, co2s, plot_kwargs, plot_legend=True):
    fig, ax = plt.subplots()
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("CO2 (ppm)")

    for t, co2, kwargs in zip(ts, co2s, plot_kwargs):
        ax.plot(t, co2, **kwargs)
    if plot_legend:
        ax.legend()
    return fig, ax


class COMP_DOMAIN:
    padding_metres = 2e3
    LAT_TO_METRES = 111e3
    points_gdf: GeoDataFrame = gpd.read_file(Path("data/locations/locations.shp"))
    min_lat = points_gdf["Latitude"].min() - padding_metres / LAT_TO_METRES
    max_lat = points_gdf["Latitude"].max() + padding_metres / LAT_TO_METRES
    min_lon = points_gdf["Longitude"].min() - padding_metres / LAT_TO_METRES
    max_lon = points_gdf["Longitude"].max() + padding_metres / LAT_TO_METRES


def get_map_axes(ax: Axes) -> Axes:
    ax.set_aspect("equal")
    ax.set_axis_off()
    ax.add_artist(ScaleBar(1, location="lower right"))
    return ax


def export_fig(fig: Figure, path: Path):
    fig.savefig(
        path.with_suffix(".svg"), bbox_inches="tight", pad_inches=None, transparent=True
    )


def plot_with_scale(
    ax: Axes, gdf_list: List[GeoDataFrame], gdf_plot_kwargs=None
) -> Axes:
    for gdf, plot_kwargs in zip(gdf_list, gdf_plot_kwargs):
        ax = gdf.to_crs(32619).plot(ax=ax, **plot_kwargs)

    ax.autoscale(tight=True)
    cx.add_basemap(
        ax,
        crs=gdf.to_crs(32619).crs.to_string(),
        source=cx.providers.OpenStreetMap.Mapnik,
    )
    ax.set_xlim(ax.set_xlim()[::-1])
    ax.set_ylim(ax.get_ylim()[::-1])
    ax = get_map_axes(ax)
    return ax
