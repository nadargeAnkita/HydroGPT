import pandas as pd
import numpy as np
import os
import joblib
from statsmodels.tsa.stattools import adfuller


# PATH HANDLING 


BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
DATA_DIR = os.path.join(BASE_DIR, "data")

def create_clean_dataset():

    print("=== Loading Raw Datasets ===")

    inflow = pd.read_csv(os.path.join(DATA_DIR, "inflow_data.csv"), parse_dates=["date"])
    upstream = pd.read_csv(os.path.join(DATA_DIR, "upstream_outflow.csv"), parse_dates=["date"])
    rainfall = pd.read_csv(os.path.join(DATA_DIR, "rainfall_data.csv"), parse_dates=["date"])
    temp = pd.read_csv(os.path.join(DATA_DIR, "temperature_data.csv"), parse_dates=["date"])
    reservoir = pd.read_csv(os.path.join(DATA_DIR, "reservoir_level.csv"), parse_dates=["date"])

   
    # MERGING DATASETS
   

    df = inflow.merge(upstream, on="date") \
               .merge(rainfall, on="date") \
               .merge(temp, on="date") \
               .merge(reservoir, on="date")

    df = df.sort_values("date").drop_duplicates("date")
    df = df.set_index("date")

   
    # HANDLE MISSING VALUES
   

    df = df.sort_index()
    df = df.interpolate(method="time")
    df = df.ffill().bfill()

  
    # FEATURE ENGINEERING
   

    # Inflow lags
    df["inflow_lag1"] = df["inflow_cumecs"].shift(1)
    df["inflow_lag3"] = df["inflow_cumecs"].shift(3)
    df["inflow_lag7"] = df["inflow_cumecs"].shift(7)

    # Rainfall lags
    df["rain_lag1"] = df["rainfall_mm"].shift(1)
    df["rain_lag3"] = df["rainfall_mm"].shift(3)

    # Rolling rainfall
    df["rain_3day"] = df["rainfall_mm"].rolling(3).sum()
    df["rain_5day"] = df["rainfall_mm"].rolling(5).sum()

    # Moving averages
    df["inflow_ma3"] = df["inflow_cumecs"].rolling(3).mean()
    df["inflow_ma7"] = df["inflow_cumecs"].rolling(7).mean()

    # Cyclical date encoding
    df["month"] = df.index.month
    df["dayofyear"] = df.index.dayofyear

    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    df["day_sin"] = np.sin(2 * np.pi * df["dayofyear"] / 365)
    df["day_cos"] = np.cos(2 * np.pi * df["dayofyear"] / 365)

    # Drop NA rows created by lag/rolling
    df = df.dropna()

   
    # STATIONARITY TEST
   

    result = adfuller(df["inflow_cumecs"])
    print("\n=== ADF Test for Stationarity ===")
    print("ADF Statistic:", result[0])
    print("p-value:", result[1])

   
    # SAVE FEATURE LIST
   

    TARGET = "inflow_cumecs"
    FEATURES = [col for col in df.columns if col != TARGET]

    joblib.dump(FEATURES, os.path.join(DATA_DIR, "feature_list.pkl"))

  
    # SAVE CLEAN DATASET
    

    output_path = os.path.join(DATA_DIR, "merged_cleaned_dataset.csv")
    df.to_csv(output_path)

    print("\nDataset created successfully!")
    print("Saved to:", output_path)
    print("Shape:", df.shape)
    print("Total Features:", len(FEATURES))


if __name__ == "__main__":
    create_clean_dataset()
