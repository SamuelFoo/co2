import sqlite3

import pandas as pd

from project import CO2_DATABASE_PATH


def delete_timeseries_data_table():
    conn = sqlite3.connect(CO2_DATABASE_PATH)
    cursor = conn.cursor()

    cursor.execute("DROP TABLE IF EXISTS timeseries_data")
    conn.commit()
    conn.close()


def create_timeseries_data_table():
    conn = sqlite3.connect(CO2_DATABASE_PATH)
    cursor = conn.cursor()

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS timeseries_data (
            id INTEGER PRIMARY KEY,
            pi TEXT,
            date TEXT,
            time TEXT,
            co2_mean REAL,
            co2_std REAL,
            temperature_mean REAL,
            temperature_std REAL,
            humidity_mean REAL,
            humidity_std REAL,
            pressure_mean REAL,
            pressure_std REAL,
            gas_mean REAL,
            gas_std REAL,
            UNIQUE(pi, date, time)
        )
        """
    )
    conn.commit()
    conn.close()


def add_data_to_timeseries_data_table(data: dict):
    conn = sqlite3.connect(CO2_DATABASE_PATH)
    cursor = conn.cursor()

    try:
        cursor.execute(
            """
            INSERT INTO timeseries_data (
                pi, date, time, co2_mean, co2_std, 
                temperature_mean, temperature_std, 
                humidity_mean, humidity_std, pressure_mean, 
                pressure_std, gas_mean, gas_std
            )
            VALUES (
                :pi, :date, :time, :co2_mean, :co2_std, 
                :temperature_mean, :temperature_std, 
                :humidity_mean, :humidity_std, :pressure_mean, 
                :pressure_std, :gas_mean, :gas_std
            )
            """,
            data,
        )
    except sqlite3.IntegrityError:
        cursor.execute(
            """
            UPDATE timeseries_data
            SET
                co2_mean = :co2_mean,
                co2_std = :co2_std,
                temperature_mean = :temperature_mean,
                temperature_std = :temperature_std,
                humidity_mean = :humidity_mean,
                humidity_std = :humidity_std,
                pressure_mean = :pressure_mean,
                pressure_std = :pressure_std,
                gas_mean = :gas_mean,
                gas_std = :gas_std
            WHERE
                pi = :pi AND date = :date AND time = :time
            """,
            data,
        )

    conn.commit()
    conn.close()


def get_co2_data_by_date(date: str) -> pd.DataFrame:
    # Connect to the database
    conn = sqlite3.connect(CO2_DATABASE_PATH)
    cursor = conn.cursor()

    # Query to get the data from the database
    query = f"""
    SELECT date, time, pi, co2_mean, co2_std
    FROM timeseries_data
    WHERE pi IN ('PI3', 'PI4') AND date = '{date}'
    ORDER BY date, time
    """

    # Load the data into a DataFrame
    df = pd.read_sql_query(query, conn)

    return df
