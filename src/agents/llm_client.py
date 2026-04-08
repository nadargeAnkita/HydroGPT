import os

# -----------------------------------------------
# Groq model priority list — tries each in order
# Falls back to next if decommissioned/unavailable
# -----------------------------------------------
GROQ_MODELS = [
    "llama-3.1-8b-instant",    # Current recommended free model
    "llama-3.3-70b-versatile", # Larger, more capable
    "mixtral-8x7b-32768",      # Mixtral fallback
    "gemma2-9b-it",            # Google Gemma fallback
]


def call_llm(messages):
    """
    Calls LLM using Groq API. Tries multiple models automatically.
    Falls back to smart rule-based response if no API key or all models fail.
    Get free key at: https://console.groq.com
    """
    api_key = os.getenv("GROQ_API_KEY")
    if api_key:
        return _call_groq(messages, api_key)
    else:
        return _smart_fallback(messages)


def _call_groq(messages, api_key):
    """Try each model in GROQ_MODELS until one works."""
    try:
        from groq import Groq
        client = Groq(api_key=api_key)
        last_error = None

        for model in GROQ_MODELS:
            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=0.7,
                    max_tokens=1024
                )
                print(f"[LLM] Using Groq model: {model}")
                return response.choices[0].message.content

            except Exception as e:
                err = str(e).lower()
                if any(x in err for x in ["decommissioned", "model_not_found",
                                           "invalid_request", "does not exist",
                                           "model_decommissioned"]):
                    print(f"[LLM] {model} unavailable, trying next...")
                    last_error = e
                    continue
                else:
                    print(f"[LLM] Groq error: {e}")
                    return _smart_fallback(messages)

        print(f"[LLM] All Groq models failed. Last error: {last_error}")
        return _smart_fallback(messages)

    except ImportError:
        print("[LLM] Groq not installed. Run: pip install groq")
        return _smart_fallback(messages)
    except Exception as e:
        print(f"[LLM] Unexpected error: {e}")
        return _smart_fallback(messages)


def _smart_fallback(messages):
    """
    Smart rule-based fallback — works fully offline without any API key.
    Generates detailed hydrology answers based on question keywords.
    """
    user_content = ""
    for msg in messages:
        if msg["role"] == "user":
            user_content = msg["content"]
            break

    user_lower = user_content.lower()

    # Extract risk info if embedded in message
    risk_info = ""
    if "risk analysis:" in user_lower:
        try:
            parts = user_content.split("Risk Analysis:")
            if len(parts) > 1:
                risk_info = parts[1].split("Hydrology Knowledge:")[0].strip().lower()
        except:
            pass

    # ── Flood risk responses ──────────────────────────────
    if "high flood risk" in user_lower or "high flood risk" in risk_info:
        return (
            "🚨 HIGH FLOOD RISK DETECTED\n\n"
            "Reservoir inflow is expected to significantly exceed normal levels, driven by "
            "heavy rainfall and saturated catchment conditions.\n\n"
            "RECOMMENDED ACTIONS:\n"
            "• Open spillway gates and increase outflow releases immediately\n"
            "• Alert downstream communities and disaster management authorities\n"
            "• Monitor reservoir water level every 6 hours\n"
            "• Prepare emergency response teams\n"
            "• Coordinate with upstream dam operators on release schedules"
        )

    elif "moderate risk" in user_lower or "moderate risk" in risk_info:
        return (
            "⚠️ MODERATE RISK — ELEVATED INFLOW\n\n"
            "Inflow is above normal but within manageable limits. Consistent with seasonal "
            "rainfall increase and upstream catchment saturation.\n\n"
            "RECOMMENDED ACTIONS:\n"
            "• Monitor reservoir levels every 12 hours\n"
            "• Review spillway readiness and gate schedules\n"
            "• Prepare for potential inflow increase if rainfall continues"
        )

    elif "low risk" in user_lower or "low risk" in risk_info:
        return (
            "✅ NORMAL CONDITIONS — LOW RISK\n\n"
            "Inflow levels are within expected seasonal ranges. "
            "Reservoir operations can continue as per standard schedule. "
            "No immediate action required."
        )

    # ── Topic-based responses ─────────────────────────────
    elif "rainfall" in user_lower and ("inflow" in user_lower or "affect" in user_lower):
        return (
            "Rainfall is the primary driver of reservoir inflow:\n\n"
            "1. DIRECT RUNOFF: When rainfall exceeds soil infiltration, water flows "
            "directly into rivers and the reservoir — this happens within hours.\n\n"
            "2. SOIL SATURATION: During sustained rainfall, soil becomes saturated. "
            "Nearly all additional rainfall then becomes surface runoff, dramatically "
            "increasing inflow.\n\n"
            "3. LAG EFFECT: Peak inflow occurs 1–5 days after peak rainfall depending "
            "on catchment size and slope. This is why HydroGPT uses rain_lag1, rain_lag3, "
            "rain_3day, and rain_5day features.\n\n"
            "4. MONSOON EFFECT: Sequential rainfall events prevent soil recovery, causing "
            "sustained high inflow for weeks even on relatively dry monsoon days."
        )

    elif "flood" in user_lower:
        return (
            "Flood risk in hydropower reservoirs is classified by predicted inflow:\n\n"
            "• LOW RISK (< 300 cumecs): Normal operations\n"
            "• MODERATE RISK (300–500 cumecs): Elevated monitoring, prepare spillways\n"
            "• HIGH RISK (> 500 cumecs): Emergency protocol, open gates, alert downstream\n\n"
            "Main causes of high flood risk:\n"
            "• Intense or prolonged monsoon rainfall\n"
            "• Soil saturation from previous rainfall\n"
            "• High upstream dam releases\n"
            "• Cyclonic storms causing extreme precipitation\n\n"
            "HydroGPT's ReAct agent automatically assesses risk after every forecast."
        )

    elif "lstm" in user_lower and "xgboost" in user_lower:
        return (
            "XGBoost vs LSTM for Hydropower Forecasting:\n\n"
            "XGBOOST:\n"
            "• Works on tabular data with engineered lag/rolling features\n"
            "• Builds 500 decision trees sequentially\n"
            "• Faster training, interpretable feature importance\n"
            "• Achieved R² = 0.862 in HydroGPT Phase 1\n\n"
            "LSTM:\n"
            "• Understands time sequences natively via memory gates\n"
            "• Better for long-range temporal dependencies\n"
            "• Requires more training time\n\n"
            "HydroGPT trains both and auto-selects the best model by R² score."
        )

    elif "lstm" in user_lower:
        return (
            "LSTM (Long Short-Term Memory) in HydroGPT:\n\n"
            "LSTM has 3 gates controlling information flow:\n"
            "• Forget Gate: Discards irrelevant past information\n"
            "• Input Gate: Stores relevant new information\n"
            "• Output Gate: Decides what to output at current step\n\n"
            "In HydroGPT, LSTM uses a 14-day lookback window to predict next day's inflow. "
            "Architecture: Input → LSTM(128) → Dropout(0.3) → LSTM(64) → Dense(32) → Output\n\n"
            "Why LSTM: River flow has long-term memory — rainfall from last week still "
            "affects today's inflow. LSTM captures this naturally."
        )

    elif "xgboost" in user_lower:
        return (
            "XGBoost in HydroGPT:\n\n"
            "Builds 500 decision trees sequentially — each tree fixes errors of the previous.\n\n"
            "HYPERPARAMETERS:\n"
            "• n_estimators = 500, learning_rate = 0.03\n"
            "• max_depth = 6, subsample = 0.8\n\n"
            "RESULTS: RMSE = 15.593, MAE = 12.461, R² = 0.862\n\n"
            "Best suited for structured tabular hydrological data with engineered features."
        )

    elif "sarimax" in user_lower or "sarima" in user_lower:
        return (
            "SARIMAX in HydroGPT — statistical baseline model:\n\n"
            "COMPONENTS:\n"
            "• AR: Uses past inflow values to predict future\n"
            "• I: Differencing to remove trends\n"
            "• MA: Uses past forecast errors\n"
            "• S: Captures monsoon seasonal cycles\n"
            "• X: External variables (rainfall, temperature)\n\n"
            "Parameters used: order=(2,1,2), seasonal_order=(1,1,1,7)\n\n"
            "Advantage: Fully interpretable and excellent for seasonal patterns."
        )

    elif "monsoon" in user_lower or "seasonal" in user_lower:
        return (
            "Monsoon and seasonal effects in hydropower forecasting:\n\n"
            "MONSOON (June–September):\n"
            "• 70–80% of annual reservoir inflow occurs during monsoon\n"
            "• Saturated catchments amplify runoff from even moderate rainfall\n"
            "• Highest flood risk period — aggressive gate management needed\n\n"
            "DRY SEASON (October–May):\n"
            "• Inflow drops sharply — often below 100 cumecs\n"
            "• Stored water sustains power generation\n"
            "• Conservation becomes the priority\n\n"
            "HydroGPT captures seasonality using cyclical encoding (month_sin/cos, "
            "day_sin/cos) and SARIMAX seasonal components."
        )

    elif "lag" in user_lower or "feature" in user_lower:
        return (
            "Lag features in HydroGPT capture hydrological memory effects:\n\n"
            "INFLOW LAGS:\n"
            "• inflow_lag1 = yesterday's inflow\n"
            "• inflow_lag3 = inflow 3 days ago\n"
            "• inflow_lag7 = inflow 7 days ago\n\n"
            "RAINFALL LAGS:\n"
            "• rain_lag1, rain_lag3 = delayed rainfall impact\n"
            "• rain_3day, rain_5day = cumulative soil saturation\n\n"
            "MOVING AVERAGES:\n"
            "• inflow_ma3, inflow_ma7 = smooth trend detection\n\n"
            "Without lag features, XGBoost cannot understand the time-delayed "
            "relationship between rainfall events and reservoir inflow response."
        )

    elif "rag" in user_lower or "retrieval" in user_lower or "knowledge" in user_lower:
        return (
            "RAG (Retrieval-Augmented Generation) in HydroGPT:\n\n"
            "1. KNOWLEDGE BASE: Curated hydrology text (rainfall-runoff theory, "
            "reservoir operations, flood risk, model explanations)\n\n"
            "2. VECTORIZATION: Knowledge split into chunks, converted to TF-IDF vectors\n\n"
            "3. RETRIEVAL: User question converted to vector → cosine similarity search "
            "→ top 3 most relevant chunks retrieved\n\n"
            "4. AUGMENTATION: Retrieved chunks + question = structured LLM prompt\n\n"
            "5. GENERATION: LLM generates grounded answer — no hallucination\n\n"
            "Result: All answers are based on verified hydrology knowledge."
        )

    elif "react" in user_lower or "agent" in user_lower:
        return (
            "ReAct Agent in HydroGPT (Reasoning + Acting):\n\n"
            "ITERATION 1:\n"
            "Thought: 'Run the inflow forecast'\n"
            "Action: forecast_tool() → gets predicted inflow values\n\n"
            "ITERATION 2:\n"
            "Thought: 'Assess flood risk'\n"
            "Action: risk_analysis_tool() → LOW/MODERATE/HIGH classification\n\n"
            "ITERATION 3:\n"
            "Thought: 'Retrieve domain knowledge'\n"
            "Action: retrieve_context() → relevant hydrology knowledge\n\n"
            "FINAL: All results → LLM → comprehensive natural language explanation\n\n"
            "Based on Yao et al. (2022) ReAct framework. Enables adaptive, "
            "multi-step reasoning unlike simple single-pass pipelines."
        )

    elif "data" in user_lower and ("clean" in user_lower or "process" in user_lower):
        return (
            "HydroGPT Data Preprocessing Pipeline:\n\n"
            "1. LOAD: 5 CSV files (inflow, upstream, rainfall, temperature, reservoir)\n"
            "2. MERGE: Join all datasets by date column\n"
            "3. CLEAN: Time interpolation → forward fill → backward fill\n"
            "4. VALIDATE: IQR outlier detection, ADF stationarity test\n"
            "5. ENGINEER: Create 14 lag, rolling, and cyclical features\n"
            "6. SPLIT: Train(2015-2022) / Val(2023) / Test(2024)\n\n"
            "Final dataset: 3,646 daily records, 21 columns, zero missing values."
        )

    else:
        return (
            "HydroGPT — AI-Driven Hydropower Inflow Forecasting Platform\n\n"
            "I can answer questions about:\n"
            "• How rainfall affects reservoir inflow\n"
            "• Flood risk assessment and recommended actions\n"
            "• XGBoost, LSTM, and SARIMAX forecasting models\n"
            "• Lag features and feature engineering\n"
            "• Monsoon and seasonal effects\n"
            "• RAG knowledge retrieval system\n"
            "• ReAct intelligent agent framework\n"
            "• Data preprocessing and cleaning\n\n"
            "Ask me anything about hydropower forecasting!"
        )
