import os
import joblib
from numpy.polynomial import test
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from statsmodels.tsa.statespace.sarimax import SARIMAX


# Paths

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
DATA_PATH = os.path.join(BASE_DIR, "data", "merged_cleaned_dataset.csv")
MODEL_DIR = os.path.join(BASE_DIR, "models")

os.makedirs(MODEL_DIR, exist_ok=True)



# Train SARIMAX (Pure Time-Series)


def train_sarimax():

    print("=== Loading Dataset ===")

    df = pd.read_csv(DATA_PATH, parse_dates=["date"], index_col="date")

    TARGET = "inflow_cumecs"

    # Ensure daily frequency
    df = df.asfreq("D")

    train = df.loc["2015":"2023"]
    test = df.loc["2024":]

    y_train = train[TARGET]
    y_test = test[TARGET]

    FEATURES = df.columns.drop("inflow_cumecs").tolist()

    X_train = train[FEATURES]
    X_test = test[FEATURES]

    print("=== Training SARIMA Model ===")

    model = SARIMAX(
        y_train,
        exog=X_train,
        order=(2, 1, 2),
        seasonal_order=(1, 1, 1, 7),
        enforce_stationarity=False,
        enforce_invertibility=False
    )

    results = model.fit(disp=False)

    
    # Predictions
   

    preds = results.forecast(steps=len(y_test), exog=X_test)

    
    # Metrics
  

    rmse = np.sqrt(mean_squared_error(y_test, preds))
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)

    print("\n=== SARIMA Evaluation ===")
    print(f"RMSE: {rmse:.3f}")
    print(f"MAE: {mae:.3f}")
    print(f"R² Score: {r2:.3f}")

    
    # Save Model
    

    joblib.dump(results, os.path.join(MODEL_DIR, "model_sarimax.pkl"))

    metrics = {
        "RMSE": rmse,
        "MAE": mae,
        "R2": r2
    }

    joblib.dump(metrics, os.path.join(MODEL_DIR, "metrics_sarimax.pkl"))

    
    # Plot
    

    plt.figure(figsize=(12, 6))
    plt.plot(y_test.index, y_test, label="Actual")
    plt.plot(y_test.index, preds, label="Predicted")
    plt.title("SARIMA - Actual vs Predicted (2024)")
    plt.xlabel("Date")
    plt.ylabel("Inflow (cumecs)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(MODEL_DIR, "sarimax_actual_vs_predicted.png"))
    plt.show()

    print("\nSARIMA training completed successfully!")


if __name__ == "__main__":
    train_sarimax()