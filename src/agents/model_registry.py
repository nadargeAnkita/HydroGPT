"""
Model Registry — loads saved model metrics and selects the best model.
Handles both metrics_xgb.pkl and metrics_xgboost.pkl naming conventions.
"""

import os
import joblib

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
MODEL_DIR = os.path.join(BASE_DIR, "models")


def load_metrics():
    """Load metrics for all available trained models."""
    metrics = {}

    # XGBoost — handle both naming conventions
    for name in ["metrics_xgb.pkl", "metrics_xgboost.pkl"]:
        path = os.path.join(MODEL_DIR, name)
        if os.path.exists(path):
            metrics["xgboost"] = joblib.load(path)
            break

    # LSTM
    path = os.path.join(MODEL_DIR, "metrics_lstm.pkl")
    if os.path.exists(path):
        metrics["lstm"] = joblib.load(path)

    # SARIMAX
    path = os.path.join(MODEL_DIR, "metrics_sarimax.pkl")
    if os.path.exists(path):
        metrics["sarimax"] = joblib.load(path)

    return metrics


def get_best_model():
    """
    Select the best model based on lowest RMSE.
    Falls back to xgboost if no metrics files found.
    """
    metrics = load_metrics()

    if not metrics:
        # Default fallback — xgboost is always trained first
        print("Warning: No metrics files found. Defaulting to xgboost.")
        return "xgboost"

    best_model = min(metrics, key=lambda m: metrics[m].get("RMSE", float("inf")))
    return best_model


def get_all_metrics():
    """Return metrics for all models as a formatted summary."""
    metrics = load_metrics()

    if not metrics:
        return "No trained models found."

    lines = ["Model Performance Summary:", "-" * 40]
    for model, m in metrics.items():
        lines.append(
            f"{model.upper()}: RMSE={m.get('RMSE', 'N/A'):.3f}, "
            f"MAE={m.get('MAE', 'N/A'):.3f}, "
            f"R²={m.get('R2', 'N/A'):.3f}"
        )

    return "\n".join(lines)
