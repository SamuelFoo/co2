import cv2
import geopandas as gpd
import numpy as np
import rasterio
from rasterio import features
from scipy.ndimage import distance_transform_edt
from shapely.geometry import LineString


def map_to_bounds(pts: np.ndarray, old_bounds, new_bounds):
    pts = pts.copy()
    for i in range(len(old_bounds)):
        scale = (new_bounds[i][1] - new_bounds[i][0]) / (
            old_bounds[i][1] - old_bounds[i][0]
        )
        pts[:, i] = (pts[:, i] - old_bounds[i][0]) * scale + new_bounds[i][0]
    return pts


def road_shapefile_to_opencv_mat(
    geo_data: gpd.GeoDataFrame, img_width: int
) -> cv2.typing.MatLike:
    """Plot shapefile line_data on an OpenCV image.

    Args:
        geo_data (gpd.GeoDataFrame): GeoDataFrame object.
        img_width (int): Image width.

    Returns:
        cv2.typing.MatLike: Image with line_data plotted.
    """
    line_data = geo_data.explode()
    coords = line_data.geometry.apply(lambda geom: list(geom.coords))

    geo_data_x_bounds = [geo_data.total_bounds[0], geo_data.total_bounds[2]]
    geo_data_y_bounds = [geo_data.total_bounds[1], geo_data.total_bounds[3]]
    geo_data_width = geo_data_x_bounds[1] - geo_data_x_bounds[0]
    geo_data_height = geo_data_y_bounds[1] - geo_data_y_bounds[0]
    img_height = round(geo_data_height / geo_data_width * img_width)
    image = np.zeros((img_height, img_width), dtype=np.uint8)

    for polyline in coords:
        pts = np.array(polyline)
        pts = map_to_bounds(
            pts,
            [geo_data_x_bounds, geo_data_y_bounds],
            [(0, img_width), (0, img_height)],
        )
        image = cv2.polylines(
            image, np.int32([pts]), isClosed=False, color=255, thickness=1
        )

    return image


def distance_transform_scipy(gdf: gpd.GeoDataFrame, resolution: int):
    """_summary_

    Args:
        gdf (gpd.GeoDataFrame): _description_
        resolution (int): Pixel size in the same units as your shapefile.

    Returns:
        _type_: _description_
    """
    bounds = gdf.total_bounds  # (minx, miny, maxx, maxy)

    # Define the raster's transform and dimensions based on the shapefile's bounds
    transform = rasterio.transform.from_bounds(
        *bounds,
        width=int((bounds[2] - bounds[0]) / resolution),
        height=int((bounds[3] - bounds[1]) / resolution)
    )
    out_shape = (
        int((bounds[3] - bounds[1]) / resolution),
        int((bounds[2] - bounds[0]) / resolution),
    )

    # Rasterize the line geometries using only LineStrings
    raster = np.zeros(out_shape, dtype=np.uint8)
    geometries = [(geom, 1) for geom in gdf.geometry if isinstance(geom, LineString)]
    raster = features.rasterize(
        geometries, out_shape=out_shape, transform=transform, fill=0, dtype=np.uint8
    )

    # Return the distance transform
    return transform, distance_transform_edt(1 - raster)
