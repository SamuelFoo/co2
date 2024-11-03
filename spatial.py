from io import BytesIO
from typing import Tuple

import geopandas as gpd
import numpy as np
import pandas as pd
from geopandas import GeoDataFrame
from scipy.spatial.distance import cdist


def get_traffic_df_from_gdf(traffic_gdf: GeoDataFrame, site_id: str):
    # Filter attachments that starts with the site_id
    site_gdf = traffic_gdf[traffic_gdf["ATT_NAME"].str.startswith(site_id)]

    # Get date from attachment name
    dates = site_gdf["ATT_NAME"].str.extract(r"(\d{4}-\d{2}-\d{2})")

    # Get data for latest date
    latest_date = dates.max().values[0]
    latest_date_row = site_gdf[dates[0] == latest_date]
    data = latest_date_row["DATA"].values[0]

    # Debug: save to xlsx
    with open("data.xlsx", "wb") as f:
        f.write(data)

    # Parse data as xslx
    xl = pd.ExcelFile(BytesIO(data))
    df = xl.parse("Data")
    return df


def get_traffic_data_from_df(df: pd.DataFrame):
    time_range = df.iloc[:, 1]
    time = time_range.str.split(" - ", expand=True)[0]

    # Set start and end rows such that they are 4 digit numbers
    start_row = time[time.str.contains(r"\d{4}", na=False)].index[0]
    end_row = time[time.str.contains(r"\d{4}", na=False)].index[-1] + 1

    time = time[start_row:end_row]
    traffic = pd.Series([0] * len(time), index=time.index)

    columns_with_totals = df.columns[
        df.apply(
            lambda x: x.astype(str).str.contains("Total").any()
            and not x.astype(str).str.contains("Time").any(),
            axis=0,
        )
    ]
    for column in columns_with_totals:
        traffic += df[column][start_row:end_row]

    time = pd.to_datetime(time, format="%H%M")
    return time, traffic


def get_traffic_data_by_site_id(traffic_gdf: GeoDataFrame, site_id: str):
    df = get_traffic_df_from_gdf(traffic_gdf, site_id)
    time, traffic = get_traffic_data_from_df(df)
    return time, traffic


def interpolate_traffic_data(time: pd.Series, traffic: pd.Series) -> pd.DataFrame:
    df = pd.DataFrame({"time": time.copy(), "traffic": traffic.astype(np.int64)})

    start_time = pd.Timestamp("1900-01-01 06:00:00")
    end_time = pd.Timestamp("1900-01-01 20:45:00")
    full_range = pd.date_range(start=start_time, end=end_time, freq="15min")
    df.set_index("time", inplace=True)
    df = df.reindex(full_range)
    df = df.resample("15min").interpolate(method="linear", limit_direction="both")
    return df


def get_traffic_count_score(
    time: str,
    gdf: GeoDataFrame,
    data_dict: dict[str, pd.Series],
    n: float = 1.0,
    k: float = 1.0,
    b: float = 0.0,
    grid_dimension: int = 100,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    timestamp = pd.to_datetime(time, format="%H%M")

    # Add a new column to traffic_gdf with interpolated traffic data
    gdf["interpolated_traffic"] = gdf["SiteID"].apply(
        lambda site_id: data_dict.get(site_id, pd.DataFrame())
        .get("traffic", pd.Series())
        .get(timestamp, 0)
    )

    transformed_gdf = gdf.to_crs("EPSG:32619")

    x: np.ndarray = transformed_gdf.geometry.x
    y: np.ndarray = transformed_gdf.geometry.y
    xi, yi = np.meshgrid(
        np.linspace(x.min(), x.max(), grid_dimension),
        np.linspace(y.min(), y.max(), grid_dimension),
    )
    pts = np.vstack([x, y]).T
    mesh_pts = np.vstack([xi.flatten(), yi.flatten()]).T

    dists = cdist(mesh_pts, pts, "euclidean")
    z = transformed_gdf["interpolated_traffic"].values

    a = np.sort(np.unique(cdist(pts, pts, "euclidean")))[1]
    diffusion = k * 1 / (a + dists) ** n + b
    traffic_score = diffusion @ z
    return xi, yi, traffic_score


def get_traffic_counts_by_time(
    time: str, gdf: GeoDataFrame, data_dict: dict[str, pd.Series]
) -> GeoDataFrame:
    timestamp = pd.to_datetime(time, format="%H%M")

    gdf["interpolated_traffic"] = gdf["SiteID"].apply(
        lambda site_id: data_dict.get(site_id, pd.DataFrame())
        .get("traffic", pd.Series())
        .get(timestamp, 0)
    )
    return gdf
