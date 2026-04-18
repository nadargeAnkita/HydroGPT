import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# ===============================
# Paths
# ===============================
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

LOOKBACK = 14


def create_sequences(X, y, lookback):
    Xs, ys = [], []                                         # empty input and output lists          
    for i in range(len(X) - lookback):                      # l = 100, lb = 14 window = 0 - 85
        Xs.append(X[i:(i + lookback)])
        ys.append(y[i + lookback])                          # 
    return np.array(Xs), np.array(ys)


def train_lstm():

    print("=== Loading Dataset ===")

    df = pd.read_csv(
        os.path.join(DATA_DIR, "merged_cleaned_dataset.csv"),
        parse_dates=["date"],
        index_col="date"
    )

    TARGET   = "inflow_cumecs"
    FEATURES = [c for c in df.columns if c != TARGET]

    # ── Splits ──────────────────────────────
    train = df.loc["2015":"2022"]
    val   = df.loc["2023"]
    test  = df.loc["2024":]

    X_train = train[FEATURES].values
    y_train = train[TARGET].values

    X_val   = val[FEATURES].values
    y_val   = val[TARGET].values

    X_test  = test[FEATURES].values
    y_test  = test[TARGET].values

    # ── Scale ────────────────────────────────
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()

    X_train_sc = scaler_X.fit_transform(X_train)
    X_val_sc   = scaler_X.transform(X_val)
    X_test_sc  = scaler_X.transform(X_test)

    y_train_sc = scaler_y.fit_transform(y_train.reshape(-1, 1))
    y_val_sc   = scaler_y.transform(y_val.reshape(-1, 1))
    y_test_sc  = scaler_y.transform(y_test.reshape(-1, 1))

    # Save scalers
    joblib.dump(scaler_X, os.path.join(MODEL_DIR, "scaler_lstm_X.pkl"))
    joblib.dump(scaler_y, os.path.join(MODEL_DIR, "scaler_lstm_y.pkl"))
    joblib.dump(FEATURES, os.path.join(MODEL_DIR, "features_lstm.pkl"))

    # ── Sequences ────────────────────────────
    X_train_seq, y_train_seq = create_sequences(X_train_sc, y_train_sc, LOOKBACK)
    X_val_seq,   y_val_seq   = create_sequences(X_val_sc,   y_val_sc,   LOOKBACK)
    X_test_seq,  y_test_seq  = create_sequences(X_test_sc,  y_test_sc,  LOOKBACK)

    print(f"Train sequences : {X_train_seq.shape}")
    print(f"Val sequences   : {X_val_seq.shape}")
    print(f"Test sequences  : {X_test_seq.shape}")

    # ── Build Model ──────────────────────────
    print("\n=== Building LSTM Model ===")

    model = Sequential([
        LSTM(64, return_sequences=True,
             input_shape=(LOOKBACK, X_train_seq.shape[2])),
        Dropout(0.2),
        LSTM(32),
        Dropout(0.2),
        Dense(16, activation="relu"),
        Dense(1)
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss="mse")
    model.summary()

    # ── Callbacks ────────────────────────────
    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=15,
        restore_best_weights=True,
        verbose=1
    )

    reduce_lr = ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=7,
        verbose=1
    )

    # ── Train ────────────────────────────────
    print("\n=== Training LSTM ===")

    history = model.fit(
        X_train_seq, y_train_seq,
        validation_data=(X_val_seq, y_val_seq),
        epochs=150,
        batch_size=32,
        callbacks=[early_stop, reduce_lr],
        verbose=1
    )

    # ── Save Model ───────────────────────────
    model.save(os.path.join(MODEL_DIR, "model_lstm.keras"))

    # ── Evaluate ─────────────────────────────
    preds_sc     = model.predict(X_test_seq)
    preds        = scaler_y.inverse_transform(preds_sc)
    y_test_actual = scaler_y.inverse_transform(y_test_seq)

    rmse = np.sqrt(mean_squared_error(y_test_actual, preds))
    mae  = mean_absolute_error(y_test_actual, preds)
    r2   = r2_score(y_test_actual, preds)

    print("\n=== LSTM Evaluation ===")
    print(f"RMSE    : {rmse:.3f}")
    print(f"MAE     : {mae:.3f}")
    print(f"R² Score: {r2:.3f}")

    # ── Save Metrics ─────────────────────────
    metrics = {"RMSE": float(rmse), "MAE": float(mae), "R2": float(r2)}
    joblib.dump(metrics, os.path.join(MODEL_DIR, "metrics_lstm.pkl"))

    # ── Plot ─────────────────────────────────
    plt.figure(figsize=(12, 5))
    plt.plot(test.index[LOOKBACK:], y_test_actual, label="Actual")
    plt.plot(test.index[LOOKBACK:], preds,          label="Predicted")
    plt.title("LSTM — Actual vs Predicted Inflow (2024)")
    plt.xlabel("Date")
    plt.ylabel("Inflow (cumecs)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(MODEL_DIR, "lstm_actual_vs_predicted.png"))
    plt.close()

    # ── Training Loss Plot ────────────────────
    plt.figure(figsize=(10, 4))
    plt.plot(history.history["loss"],     label="Train Loss")
    plt.plot(history.history["val_loss"], label="Val Loss")
    plt.title("LSTM Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(MODEL_DIR, "lstm_training_loss.png"))
    plt.close()

    print("\nLSTM training completed successfully!")


if __name__ == "__main__":
    train_lstm()
