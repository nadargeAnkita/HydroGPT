"""
Model Selector — selects the best available trained model based on R² score.
"""

import os
import joblib

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
MODEL_DIR = os.path.join(BASE_DIR, "models")


def load_metrics_safe(model_name):
    """Load metrics for a model, trying multiple filename conventions."""

    candidates = [
        os.path.join(MODEL_DIR, f"metrics_{model_name}.pkl"),
        os.path.join(MODEL_DIR, f"metrics_xgb.pkl") if model_name == "xgboost" else None,
    ]

    for path in candidates:
        if path and os.path.exists(path):
            try:
                return joblib.load(path)
            except Exception:
                continue

    return None


def get_best_model():
    """
    Select best model by highest R² score.
    Returns (model_name, r2_score).
    Falls back to xgboost if no metrics found.
    """

    models = ["xgboost", "lstm", "sarimax"]
    best_model = None
    best_r2 = -999

    for m in models:
        metrics = load_metrics_safe(m)
        if metrics is None:
            continue
        r2 = metrics.get("R2", -999)
        if r2 > best_r2:
            best_r2 = r2
            best_model = m

    if best_model is None:
        print("Warning: No trained model metrics found. Defaulting to xgboost.")
        return "xgboost", 0.0

    return best_model, best_r2
