import sqlite3
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from filterpy.kalman import MerweScaledSigmaPoints, UnscentedKalmanFilter
from natsort import natsorted
from numpy.typing import ArrayLike

from database import add_data_to_timeseries_data_table
from project import CO2_DATABASE_PATH, DATA_DIR


def ukf_rts_smooth(t, y, R=1000, x0=None):
    sigmas = MerweScaledSigmaPoints(n=1, alpha=1e-3, beta=2, kappa=0.0)
    ukf = UnscentedKalmanFilter(
        dim_x=1,
        dim_z=1,
        dt=np.gradient(t).mean(),
        hx=lambda x: x,
        fx=lambda x, dt: x,
        points=sigmas,
    )

    if x0 is None:
        ukf.x = np.array([y[0]])
    else:
        ukf.x = np.array([x0])

    ukf.R *= R

    mu, cov = ukf.batch_filter(zs=y, dts=np.gradient(t))
    xs, Ps, Ks = ukf.rts_smoother(mu, cov)
    return xs


def get_mean_std_from_timeseries(
    df: pd.DataFrame, column_name: str, start_after_time: float = 100
):
    x = df[column_name][df["Time"] > start_after_time]
    return x.mean(), x.std()


def process_timeseries(
    date: str,
    pi: str,
    parameters: dict[Tuple, dict],
    default_R=1000,
    override_show_plot=None,
    cuttoff=1000,
):
    data_dir = DATA_DIR / f"{date}/{pi}"

    for file_path in natsorted(data_dir.glob("*.csv")):
        timeseries = str(file_path.relative_to(DATA_DIR / date).with_suffix(""))
        date = timeseries.split("/")[1].split(" ")[0]
        time = "".join(timeseries.split("/")[1].split(" ")[1].split("_")[:2]).strip("_")
        pi = timeseries.split("/")[0]

        x0 = None
        R = default_R
        show_plot = True
        x_start = x_end = None
        if (date, time) in parameters:
            parameter_dict = parameters[(date, time)]
            if "x0" in parameter_dict:
                x0 = parameter_dict["x0"]
            if "R" in parameter_dict:
                R = parameter_dict["R"]
            if "show" in parameter_dict:
                show_plot = parameter_dict["show"]
            x_start = parameter_dict.get("x_start", None)
            x_end = parameter_dict.get("x_end", None)

        if override_show_plot is not None:
            show_plot = override_show_plot

        file_path = DATA_DIR / date / f"{timeseries}.csv"
        df = pd.read_csv(file_path)
        t, co2 = df["Time"], df["CO2"]
        cuttoff_mask = co2 < cuttoff
        t = t[cuttoff_mask]
        co2 = co2[cuttoff_mask]

        co2_smoothed = ukf_rts_smooth(t, co2, x0=x0, R=R)

        if show_plot:
            plt.plot(t, co2, label=timeseries)
            plt.plot(t, co2_smoothed)
            if x_start is not None:
                plt.axvline(x_start, color="black", linestyle="--")
            if x_end is not None:
                plt.axvline(x_end, color="black", linestyle="-.")
            plt.legend()
            plt.show()

        mask = t > x_start if x_start is not None else np.ones_like(t, dtype=bool)
        mask &= t < x_end if x_end is not None else np.ones_like(t, dtype=bool)
        co2_smoothed: np.ndarray = co2_smoothed[mask]
        mean = co2_smoothed.mean()
        std = co2_smoothed.std()

        temperature_mean, temperature_std = get_mean_std_from_timeseries(
            df, "Temperature"
        )
        pressure_mean, pressure_std = get_mean_std_from_timeseries(df, "Pressure")
        humidity_mean, humidity_std = get_mean_std_from_timeseries(df, "Humidity")
        gas_mean, gas_std = get_mean_std_from_timeseries(df, "Gas")

        data = {
            "pi": pi,
            "date": date,
            "time": time,
            "co2_mean": mean,
            "co2_std": std,
            "temperature_mean": temperature_mean,
            "temperature_std": temperature_std,
            "pressure_mean": pressure_mean,
            "pressure_std": pressure_std,
            "humidity_mean": humidity_mean,
            "humidity_std": humidity_std,
            "gas_mean": gas_mean,
            "gas_std": gas_std,
        }

        add_data_to_timeseries_data_table(data)


def convert_pi4_to_pi3_co2(co2: ArrayLike) -> ArrayLike:
    """Converts CO2 value(s) from PI4 to PI3.
    See https://www.symbolab.com/solver/equation-calculator/x-y%3Da%5Cleft(x%2By%5Cright)%2F2%2Bb?or=input.

    Args:
        co2 (ArrayLike): PI4 CO2 value(s).

    Returns:
        ArrayLike: PI3 CO2 value(s).
    """
    a = 1.495635205566287  # Slope
    b = -743.3423745592194  # Intercept
    corrected_co2 = ((a + 2) * co2 + 2 * b) / (2 - a)
    return np.clip(corrected_co2, a_min=co2 - 60, a_max=co2 + 20)


def get_corrected_data() -> pd.DataFrame:
    """Corrects the CO2 values in the DataFrame from PI4 to PI3.

    Args:
        df (pd.DataFrame): DataFrame with CO2 values from PI4.

    Returns:
        pd.DataFrame: DataFrame with corrected CO2 values.
    """
    # Connect to the database
    conn = sqlite3.connect(CO2_DATABASE_PATH)

    # Query to get all data from the timeseries_data table
    query = "SELECT * FROM timeseries_data"

    # Execute the query and fetch all data
    df = pd.read_sql_query(query, conn)

    # Close the connection
    conn.close()

    # Display the dataframe
    df_corrected = df.copy()
    df_corrected.loc[df["pi"] == "PI4", "co2_mean"] = convert_pi4_to_pi3_co2(
        df_corrected["co2_mean"][df["pi"] == "PI4"]
    )
    return df
