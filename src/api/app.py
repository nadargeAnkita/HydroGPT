"""
HydroGPT FastAPI Backend
Report Section: 3.1 Sequence Diagram — FastAPI server
"""

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.agents.model_selector import get_best_model
from src.agents.model_registry import load_metrics, get_all_metrics
from src.rag.retriever import retrieve_context
from src.agents.llm_client import call_llm

# ===============================
# FastAPI App
# ===============================

app = FastAPI(
    title="HydroGPT API",
    description="AI-Driven Hydropower Inflow Forecasting Platform",
    version="1.0.0"
)

# ===============================
# Request Schemas
# ===============================

class ForecastRequest(BaseModel):
    start_date: str
    end_date: str

class ChatRequest(BaseModel):
    question: str

# ===============================
# Health Check
# ===============================

@app.get("/")
def root():
    return {
        "system": "HydroGPT",
        "status": "running",
        "version": "1.0.0",
        "endpoints": ["/predict", "/chat", "/best-model", "/model-metrics"]
    }

# ===============================
# Endpoint: Best Model
# ===============================

@app.get("/best-model")
def best_model():
    model, r2 = get_best_model()
    return {
        "best_model": model,
        "r2_score": round(r2, 4)
    }

# ===============================
# Endpoint: All Model Metrics
# ===============================

@app.get("/model-metrics")
def model_metrics():
    metrics = load_metrics()
    best = None
    if metrics:
        best = min(metrics, key=lambda m: metrics[m].get("RMSE", float("inf")))
    return {
        "metrics": metrics,
        "best_model": best
    }

# ===============================
# Endpoint: Predict (ReAct Agent)
# ===============================

@app.post("/predict")
def predict(request: ForecastRequest):

    # Validate dates
    try:
        start = pd.to_datetime(request.start_date)
        end = pd.to_datetime(request.end_date)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD.")

    if end < start:
        raise HTTPException(status_code=400, detail="end_date must be after start_date.")

    if (end - start).days > 365:
        raise HTTPException(status_code=400, detail="Forecast range cannot exceed 365 days.")

    # Run ReAct Agent
    from src.agents.react_agent import run_forecast_agent

    try:
        result = run_forecast_agent(
            start_date=request.start_date,
            end_date=request.end_date
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Forecast failed: {str(e)}")

# ===============================
# Endpoint: HydroGPT Chat
# ===============================

@app.post("/chat")
def chat(request: ChatRequest):

    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    query = request.question

    # Retrieve domain knowledge via RAG
    context = retrieve_context(query, top_k=3)

    messages = [
        {
            "role": "system",
            "content": (
                "You are HydroGPT, an expert in hydrology, reservoir inflow forecasting, "
                "flood risk analysis, and hydropower operations. "
                "Answer clearly and concisely using the provided context."
            )
        },
        {
            "role": "user",
            "content": f"Context:\n{context}\n\nQuestion:\n{query}"
        }
    ]

    answer = call_llm(messages)

    return {
        "question": query,
        "answer": answer,
        "context_used": context
    }
