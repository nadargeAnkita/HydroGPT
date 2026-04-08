import os
import joblib
import numpy as np
import pandas as pd


BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
MODEL_DIR = os.path.join(BASE_DIR, "models")
DATA_PATH = os.path.join(BASE_DIR, "data", "merged_cleaned_dataset.csv")

LOOKBACK = 14  # must match train_lstm.py


def predict_with_model(model_name, start_date, end_date):
    """Route prediction to the correct model."""
    if model_name == "xgboost":
        return predict_xgboost(start_date, end_date)
    elif model_name == "lstm":
        return predict_lstm(start_date, end_date)
    elif model_name == "sarimax":
        return predict_sarimax(start_date, end_date)
    else:
        raise ValueError(f"Unknown model type: {model_name}")


# ===============================
# XGBoost Prediction
# ===============================

def predict_xgboost(start_date, end_date):

    model = joblib.load(os.path.join(MODEL_DIR, "model_xgb.pkl"))
    scaler = joblib.load(os.path.join(MODEL_DIR, "scaler_xgb.pkl"))
    FEATURES = joblib.load(os.path.join(MODEL_DIR, "features_xgb.pkl"))

    df = pd.read_csv(DATA_PATH, parse_dates=["date"], index_col="date")

    df_range = df.loc[start_date:end_date]

    if df_range.empty:
        raise ValueError(f"No data available for date range {start_date} to {end_date}.")

    X = df_range[FEATURES]
    X_scaled = scaler.transform(X)
    preds = model.predict(X_scaled)

    return pd.Series(preds, index=df_range.index, name="forecast")


# ===============================
# LSTM Prediction
# ===============================

def predict_lstm(start_date, end_date):
    """
    LSTM inference using saved model and scalers.
    Uses a lookback window of 14 days before the start date.
    """

    try:
        import tensorflow as tf
        from sklearn.preprocessing import MinMaxScaler
    except ImportError:
        raise ImportError("TensorFlow is required for LSTM prediction.")

    # Load model
    model_path = os.path.join(MODEL_DIR, "model_lstm.keras")
    if not os.path.exists(model_path):
        raise FileNotFoundError("LSTM model not found. Please run train_lstm.py first.")

    model = tf.keras.models.load_model(model_path)

    # Load dataset
    df = pd.read_csv(DATA_PATH, parse_dates=["date"], index_col="date")

    TARGET = "inflow_cumecs"
    FEATURES = [c for c in df.columns if c != TARGET]

    # Get the window: LOOKBACK days before start + the requested range
    start_dt = pd.to_datetime(start_date)
    end_dt = pd.to_datetime(end_date)

    # Find the position of start_date in the index
    all_dates = df.index
    start_pos = all_dates.searchsorted(start_dt)

    if start_pos < LOOKBACK:
        raise ValueError(f"Not enough historical data before {start_date} for LSTM (need {LOOKBACK} days).")

    # Extract window: lookback days before start + forecast range
    window_start_pos = start_pos - LOOKBACK
    end_pos = all_dates.searchsorted(end_dt, side="right")

    df_window = df.iloc[window_start_pos:end_pos]

    X_window = df_window[FEATURES].values
    y_window = df_window[TARGET].values

    # Scale
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()

    X_scaled = scaler_X.fit_transform(X_window)
    y_scaled = scaler_y.fit_transform(y_window.reshape(-1, 1))

    # Build sequences for prediction range only
    forecast_dates = pd.date_range(start=start_date, end=end_date, freq="D")
    predictions = []

    for i in range(len(forecast_dates)):
        seq_start = i
        seq_end = i + LOOKBACK
        if seq_end > len(X_scaled):
            break

        X_seq = X_scaled[seq_start:seq_end].reshape(1, LOOKBACK, len(FEATURES))
        pred_scaled = model.predict(X_seq, verbose=0)
        pred = scaler_y.inverse_transform(pred_scaled)[0][0]
        predictions.append(pred)

    actual_dates = forecast_dates[:len(predictions)]

    return pd.Series(predictions, index=actual_dates, name="forecast")


# ===============================
# SARIMAX Prediction
# ===============================

def predict_sarimax(start_date, end_date):

    model_path = os.path.join(MODEL_DIR, "model_sarimax.pkl")
    if not os.path.exists(model_path):
        raise FileNotFoundError("SARIMAX model not found. Please run train_sarimax.py first.")

    model = joblib.load(model_path)

    forecast_days = (pd.to_datetime(end_date) - pd.to_datetime(start_date)).days + 1
    forecast = model.forecast(steps=forecast_days)
    forecast.index = pd.date_range(start=start_date, periods=forecast_days, freq="D")

    return forecast
