import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau



# Paths


BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")

os.makedirs(MODEL_DIR, exist_ok=True)



# Sequence Creator


def create_sequences(X, y, lookback):
    Xs, ys = [], []
    for i in range(len(X) - lookback):
        Xs.append(X[i:(i + lookback)])
        ys.append(y[i + lookback])
    return np.array(Xs), np.array(ys)



# Train LSTM


def train_lstm():

    print("=== Loading Dataset ===")

    df = pd.read_csv(
        os.path.join(DATA_DIR, "merged_cleaned_dataset.csv"),
        parse_dates=["date"],
        index_col="date"
    )

    TARGET = "inflow_cumecs"
    FEATURES = df.columns.drop(TARGET).tolist()

    train_full = df.loc["2015":"2023"]
    test = df.loc["2024":]

    X_train = train_full[FEATURES].values
    y_train = train_full[TARGET].values

    X_test = test[FEATURES].values
    y_test = test[TARGET].values

    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()

    X_train = scaler_X.fit_transform(X_train)
    X_test = scaler_X.transform(X_test)

    y_train = scaler_y.fit_transform(y_train.reshape(-1,1))
    y_test_scaled = scaler_y.transform(y_test.reshape(-1,1))

    lookback = 14

    X_train_seq, y_train_seq = create_sequences(X_train, y_train, lookback)
    X_test_seq, y_test_seq = create_sequences(X_test, y_test_scaled, lookback)

    print("=== Building LSTM Model ===")

    model = Sequential([
        Input(shape=(lookback, X_train_seq.shape[2])),
        LSTM(128, return_sequences=True),
        Dropout(0.3),
        LSTM(64),
        Dropout(0.3),
        Dense(32, activation="relu"),
        Dense(1)
    ])

    model.compile(optimizer="adam", loss="mse")

    early_stop = EarlyStopping(
        monitor="loss",
        patience=10,
        restore_best_weights=True
    )

    reduce_lr = ReduceLROnPlateau(
        monitor="loss",
        factor=0.5,
        patience=5,
        verbose=1
    )

    model.fit(
        X_train_seq,
        y_train_seq,
        epochs=100,
        batch_size=32,
        callbacks=[early_stop, reduce_lr],
        verbose=1
    )

  
    # Save Model
    

    model.save(os.path.join(MODEL_DIR, "model_lstm.keras"))

    
    # Predictions
   

    preds_scaled = model.predict(X_test_seq)
    preds = scaler_y.inverse_transform(preds_scaled)
    y_test_final = scaler_y.inverse_transform(y_test_seq)

    rmse = np.sqrt(mean_squared_error(y_test_final, preds))
    mae = mean_absolute_error(y_test_final, preds)
    r2 = r2_score(y_test_final, preds)

    print("\n=== LSTM Evaluation ===")
    print(f"RMSE: {rmse:.3f}")
    print(f"MAE: {mae:.3f}")
    print(f"R² Score: {r2:.3f}")

   
    # SAVE METRICS (IMPORTANT FIX)
  

    metrics = {
        "RMSE": float(rmse),
        "MAE": float(mae),
        "R2": float(r2)
    }

    joblib.dump(metrics, os.path.join(MODEL_DIR, "metrics_lstm.pkl"))

   
    # Plot
    

    plt.figure(figsize=(12,6))
    plt.plot(test.index[lookback:], y_test_final, label="Actual")
    plt.plot(test.index[lookback:], preds, label="Predicted")
    plt.legend()
    plt.grid(True)
    plt.title("LSTM - 2024 Prediction")
    plt.tight_layout()
    plt.savefig(os.path.join(MODEL_DIR, "lstm_actual_vs_predicted.png"))
    plt.close()

    print("\nLSTM training completed successfully!")


if __name__ == "__main__":
    train_lstm()