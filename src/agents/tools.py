import os
import joblib
import pandas as pd

from src.algorithm_engine.predict import predict_with_model
from src.agents.model_selector import get_best_model


# ==============================
# Forecast Tool
# ==============================

def forecast_tool(start_date: str, end_date: str):
    """
    Runs forecasting using the best available ML model
    """

    model_name, r2 = get_best_model()

    forecast = predict_with_model(
        model_name=model_name,
        start_date=start_date,
        end_date=end_date
    )

    return {
        "tool": "forecast_tool",
        "model_used": model_name,
        "model_r2": r2,
        "forecast": forecast.to_dict()
    }


# ==============================
# Model Metrics Tool
# ==============================

def model_metrics_tool():

    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
    MODEL_DIR = os.path.join(BASE_DIR, "models")

    models = ["xgboost", "sarimax", "lstm"]

    results = {}

    for m in models:

        path = os.path.join(MODEL_DIR, f"metrics_{m}.pkl")

        if os.path.exists(path):

            metrics = joblib.load(path)

            results[m] = metrics

    return {
        "tool": "model_metrics_tool",
        "metrics": results
    }


# ==============================
# Risk Analysis Tool
# ==============================

def risk_analysis_tool(start_date: str, end_date: str):
    """
    Simple flood risk analysis based on predicted inflow
    """

    forecast_data = forecast_tool(start_date, end_date)

    values = list(forecast_data["forecast"].values())

    avg_flow = sum(values) / len(values)

    if avg_flow > 500:
        risk = "HIGH FLOOD RISK"

    elif avg_flow > 300:
        risk = "MODERATE RISK"

    else:
        risk = "LOW RISK"

    return {
        "tool": "risk_analysis_tool",
        "average_inflow": avg_flow,
        "risk_level": risk,
        "forecast": forecast_data["forecast"]
    }