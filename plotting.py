from pathlib import Path
from typing import List, Tuple

import contextily as cx
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from geopandas.geodataframe import GeoDataFrame
from matplotlib.axes import Axes
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.dates import AutoDateLocator, DateFormatter
from matplotlib.figure import Figure
from matplotlib_scalebar.scalebar import ScaleBar
from scipy.interpolate import griddata
from shapely.geometry import box


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


class VIS_DOMAIN:
    min_lon, max_lon = 103.76, 103.775
    min_lat, max_lat = 1.303, 1.3225
    x_bounds_32619 = (np.float64(1307587.3368352419), np.float64(1305900.3274759217))
    y_bounds_32619 = (np.float64(19850748.35908098), np.float64(19848571.45448904))


def get_map_axes(ax: Axes) -> Axes:
    ax.set_aspect("equal")
    ax.set_axis_off()
    ax.add_artist(ScaleBar(1, location="lower right"))
    return ax


def export_fig(fig: Figure, path: Path) -> None:
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


def crop_gdf_by_vis_domain(gdf: GeoDataFrame) -> GeoDataFrame:
    bbox = box(
        VIS_DOMAIN.min_lon, VIS_DOMAIN.min_lat, VIS_DOMAIN.max_lon, VIS_DOMAIN.max_lat
    )
    bbox_gdf = gpd.GeoDataFrame([[1]], geometry=[bbox], crs="EPSG:4326")
    transformed_gdf = gdf.to_crs(crs="EPSG:4326")
    cropped_gdf = gpd.overlay(
        transformed_gdf, bbox_gdf, how="intersection", keep_geom_type=False
    )
    return cropped_gdf


def get_vis_base_plot() -> Tuple[Figure, Axes]:
    gdf: GeoDataFrame = gpd.read_file(
        Path("gis/RoadSectionLine_Jul2024/RoadSectionLine.shp")
    )
    points_gdf: GeoDataFrame = gpd.read_file(Path("data/locations/locations.shp"))
    cropped_gdf = crop_gdf_by_vis_domain(gdf)

    fig, ax = plt.subplots()
    ax = plot_with_scale(
        ax, [points_gdf, cropped_gdf], [{"color": "red", "zorder": 3}, {"zorder": -1}]
    )
    return fig, ax


def add_vis_base_map(ax: Axes) -> None:
    cx.add_basemap(
        ax,
        crs="EPSG:32619",
        source=cx.providers.OpenStreetMap.Mapnik,
        zoom=16,
    )


def plot_route(
    points_gdf: GeoDataFrame,
    ax: Axes,
    point_size: float = 30,
    color: str = "C0",
    label=None,
) -> Axes:
    """Plot paths or arrows between points"""
    points_gdf = points_gdf.to_crs(32619)
    for i in range(len(points_gdf) - 1):
        start_point = points_gdf.iloc[i].geometry
        end_point = points_gdf.iloc[i + 1].geometry
        dx = end_point.x - start_point.x
        dy = end_point.y - start_point.y
        theta = np.arctan2(dy, dx)
        offset_x = point_size * np.cos(theta)
        offset_y = point_size * np.sin(theta)
        label = label if i == 0 else None
        ax.arrow(
            start_point.x,
            start_point.y,
            dx - offset_x,
            dy - offset_y,
            head_width=55,
            length_includes_head=True,
            ec="k",
            fc=color,
            label=label,
        )
    return ax


def get_traffic_count_axes(ax: Axes) -> Axes:
    ax.set_xlabel("Time")
    ax.set_ylabel("Traffic Count")
    date_format = DateFormatter("%H:%M")
    ax.xaxis.set_major_formatter(date_format)
    ax.xaxis.set_major_locator(AutoDateLocator())
    ax.set_xlim(pd.Timestamp("1900-01-01 06:00"), pd.Timestamp("1900-01-01 21:00"))
    ax.set_ylim(0, 1200)
    plt.xticks(rotation=45)
    return ax


def plot_contour_with_map(
    gdf: GeoDataFrame,
    value_column_name: str,
    interpolation_method: str = "linear",
    map_true_range: bool = False,
) -> None:
    # Create grid data for contour plot
    x = gdf.geometry.x
    y = gdf.geometry.y
    z = gdf[value_column_name]

    # Define grid
    xi = np.linspace(x.min(), x.max(), 500)
    yi = np.linspace(y.min(), y.max(), 500)
    xi, yi = np.meshgrid(xi, yi)

    # Interpolate data
    zi = griddata((x, y), z, (xi, yi), method=interpolation_method)

    fig, ax = plt.subplots(figsize=(10, 10))
    gdf_plot_kwargs = {
        "zorder": 3,
        "column": "interpolated_traffic",
        "cmap": "hot",
        "markersize": 50,
    }
    norm = Normalize(vmin=0, vmax=5000)
    ax = plot_with_scale(ax, [gdf], [gdf_plot_kwargs])
    contour = ax.contourf(
        xi, yi, zi, levels=14, cmap="hot", zorder=4, alpha=0.8, norm=norm
    )
    if map_true_range:
        fig.colorbar(contour, ax=ax, shrink=0.8)
    else:
        fig.colorbar(
            mappable=ScalarMappable(norm=contour.norm, cmap=contour.cmap),
            ax=ax,
            shrink=0.8,
        )
    return fig, ax
