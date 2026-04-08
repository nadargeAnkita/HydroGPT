import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor
import joblib


# ROBUST PATH HANDLING


BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")

os.makedirs(MODEL_DIR, exist_ok=True)


def train_xgboost():

    print("=== Loading Dataset ===")

    df = pd.read_csv(
        os.path.join(DATA_DIR, "merged_cleaned_dataset.csv"),
        parse_dates=["date"],
        index_col="date"
    )

    
    # Train / Validation / Test Split
   

    train = df.loc["2015":"2022"]
    val = df.loc["2023"]
    test = df.loc["2024":]

    TARGET = "inflow_cumecs"
    FEATURES = df.columns.drop(TARGET).tolist()

    X_train, y_train = train[FEATURES], train[TARGET]
    X_val, y_val = val[FEATURES], val[TARGET]
    X_test, y_test = test[FEATURES], test[TARGET]

   
    # Scaling
   

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler_xgb.pkl"))
    joblib.dump(FEATURES, os.path.join(MODEL_DIR, "features_xgb.pkl"))

   
    # XGBoost Model
    

    model = XGBRegressor(
        n_estimators=500,
        learning_rate=0.03,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric="rmse"
    )

    model.fit(
        X_train_scaled, y_train,
        eval_set=[(X_val_scaled, y_val)],
        verbose=False
    )


    joblib.dump(model, os.path.join(MODEL_DIR, "model_xgb.pkl"))

    
    # Evaluation
   

    preds = model.predict(X_test_scaled)

    rmse = np.sqrt(mean_squared_error(y_test, preds))
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)

    residuals = y_test - preds
    residual_std = np.std(residuals)
    joblib.dump(residual_std, os.path.join(MODEL_DIR, "residual_std_xgb.pkl"))

    print("\n=== Model Evaluation ===")
    print(f"RMSE: {rmse:.3f}")
    print(f"MAE: {mae:.3f}")
    print(f"R² Score: {r2:.3f}")

    metrics = {
        "RMSE": rmse,
        "MAE": mae,
        "R2": r2
    }

    joblib.dump(metrics, os.path.join(MODEL_DIR, "metrics_xgb.pkl"))

    
    # Plot Validation RMSE
    

    evals_result = model.evals_result()

    plt.figure(figsize=(10,5))
    plt.plot(evals_result['validation_0']['rmse'])
    plt.title("Validation RMSE over Iterations")
    plt.xlabel("Iteration")
    plt.ylabel("RMSE")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(MODEL_DIR, "rmse_over_iterations.png"))
    plt.close()

    
    # Plot Actual vs Predicted
    

    plt.figure(figsize=(12,6))
    plt.plot(y_test.index, y_test, label="Actual")
    plt.plot(y_test.index, preds, label="Predicted")
    plt.title("Actual vs Predicted Inflow (2024 Test Set)")
    plt.xlabel("Date")
    plt.ylabel("Inflow (cumecs)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(MODEL_DIR, "actual_vs_predicted.png"))
    plt.close()

    
    # Feature Importance
  

    from xgboost import plot_importance

    plt.figure(figsize=(8,6))
    plot_importance(model, max_num_features=15)
    plt.tight_layout()
    plt.savefig(os.path.join(MODEL_DIR, "feature_importance.png"))
    plt.close()

    print("\nXGBoost training completed successfully!")


if __name__ == "__main__":
    train_xgboost()
