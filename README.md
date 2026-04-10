# 💧 HydroGPT: A No-Code AI Agent for Hydropower Forecasting

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-blue?style=for-the-badge&logo=python" />
  <img src="https://img.shields.io/badge/XGBoost-ML%20Model-orange?style=for-the-badge" />
  <img src="https://img.shields.io/badge/LSTM-Deep%20Learning-red?style=for-the-badge" />
  <img src="https://img.shields.io/badge/FastAPI-Backend-green?style=for-the-badge&logo=fastapi" />
  <img src="https://img.shields.io/badge/Streamlit-Dashboard-ff4b4b?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Groq-LLaMA%203-purple?style=for-the-badge" />
</p>

> **HydroGPT** is a domain-specific intelligent agent for hydropower reservoir inflow forecasting. It combines **XGBoost**, **LSTM**, and **SARIMAX** forecasting models with a **ReAct reasoning agent**, **RAG knowledge base**, and **Groq LLaMA 3** — all accessible through a no-code **Streamlit dashboard**. Engineers can simply describe their goals in plain English and HydroGPT autonomously handles forecasting, risk analysis, and explanation.

---

## 📌 Table of Contents

1. [Overview](#-overview)
2. [Problem Statement](#-problem-statement)
3. [Key Features](#-key-features)
4. [System Architecture](#-system-architecture)
5. [Project Structure](#-project-structure)
6. [Technologies Used](#-technologies-used)
7. [Dataset Description](#-dataset-description)
8. [Feature Engineering](#-feature-engineering)
9. [Forecasting Models](#-forecasting-models)
10. [RAG Knowledge System](#-rag-knowledge-system)
11. [ReAct Intelligent Agent](#-react-intelligent-agent)
12. [API Endpoints](#-api-endpoints)
13. [No-Code Dashboard](#-no-code-dashboard)
14. [Installation and Setup](#-installation-and-setup)
15. [How to Run](#-how-to-run)
16. [Model Performance Results](#-model-performance-results)
17. [References](#-references)
18. [Author](#-author)

---

## 🌊 Overview

**HydroGPT** is a domain-specific intelligent agent designed to democratize hydropower reservoir inflow forecasting. It transforms complex hydrological data and machine learning workflows into a simple, conversational, no-code platform that any hydropower engineer can use — regardless of programming background.

The system combines:
- **Three ML/DL models** — XGBoost, LSTM, SARIMAX — with automatic best-model selection
- **ReAct intelligent agent** — multi-step Thought → Action → Observation → Revise loop
- **RAG knowledge base** — grounded hydrology question answering without hallucination
- **Groq LLaMA 3 LLM** — natural language explanations of forecasts and flood risk
- **FastAPI + Streamlit** — production-ready backend and no-code dashboard



---

## ❓ Problem Statement

Hydropower reservoir managers face a critical daily challenge:

> *"How much water will flow into the reservoir over the next few days?"*

Poor inflow forecasting leads to:
- **Flooding** — if high inflow is not anticipated and gates are not opened in time
- **Power shortage** — if low inflow empties the reservoir unexpectedly
- **Unsafe operations** — if operators cannot make data-driven gate decisions

Traditional forecasting requires manual data collection from multiple sources, complex preprocessing scripts, expert-level ML knowledge, and significant time investment for each forecast cycle.

**HydroGPT solves all of these** by providing a unified, automated, AI-powered platform accessible through plain English — no coding required.

---

## ✨ Key Features

| Feature | Description |
|---------|-------------|
| 🤖 **ReAct Agent** | Multi-step Thought → Action → Observe → Revise reasoning loop |
| 📊 **3 ML Models** | XGBoost, LSTM, SARIMAX — auto-selects best performer by R² |
| 🧠 **RAG System** | Retrieves verified hydrology knowledge to prevent hallucination |
| 💬 **LLM Explanations** | Plain English forecast summaries via Groq LLaMA 3 |
| 🚨 **Flood Risk Alert** | Automatic LOW / MODERATE / HIGH risk classification |
| 📈 **No-Code Dashboard** | Streamlit UI — no programming skills needed |
| ⚡ **REST API** | FastAPI backend with 4 production-ready endpoints |
| 🔁 **Smart Fallback** | Fully functional without API key using rule-based LLM |
| 📥 **CSV Export** | Download forecast data for external planning tools |
| 🔍 **Reasoning Trace** | View the agent's step-by-step thinking for transparency |

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    USER (Hydropower Engineer)                   │
│              Types request in plain English / clicks UI         │
└──────────────────────────────┬──────────────────────────────────┘
                               │
┌──────────────────────────────▼──────────────────────────────────┐
│               LAYER 6 — Streamlit No-Code Dashboard             │
│        Tab 1: Forecast  |  Tab 2: Chat  |  Tab 3: Models        │
└──────────────────────────────┬──────────────────────────────────┘
                               │ HTTP Request
┌──────────────────────────────▼──────────────────────────────────┐
│              LAYER 5 — FastAPI Backend                          │
│       /predict   /chat   /best-model   /model-metrics           │
└──────────────────────────────┬──────────────────────────────────┘
                               │
┌──────────────────────────────▼──────────────────────────────────┐
│              LAYER 4 — ReAct Intelligent Agent                  │
│                                                                 │
│   Thought ──► Action ──► Observation ──► Revise ──► Repeat      │
│                   │                                             │
│              ┌────▼─────────────────────┐                       │
│              │  Tools                   │                       │
│              │  • forecast_tool()       │                       │
│              │  • risk_analysis_tool()  │                       │
│              │  • retrieve_context()    │                       │
│              └──────────────────────────┘                       │
└──────┬────────────────────────────────────┬─────────────────────┘
       │                                    │
┌──────▼───────────┐            ┌───────────▼─────────────────────┐
│  LAYER 3         │            │  LAYER 3b — RAG + LLM           │
│  ML Models       │            │                                 │
│  ┌─────────────┐ │            │  knowledge_base.txt             │
│  │  XGBoost    │ │            │       ↓                         │
│  │  R²=0.862   │ │            │  TF-IDF Vector Store            │
│  └─────────────┘ │            │       ↓                         │
│  ┌─────────────┐ │            │  Cosine Similarity Search       │
│  │    LSTM     │ │            │       ↓                         │
│  │  14-day     │ │            │  Retrieved Context              │
│  └─────────────┘ │            │       ↓                         │
│  ┌─────────────┐ │            │  Groq LLaMA 3 / Fallback        │
│  │  SARIMAX    │ │            │       ↓                         │
│  │  Seasonal   │ │            │  Natural Language Answer        │
│  └─────────────┘ │            └─────────────────────────────────┘
└──────┬───────────┘
       │
┌──────▼───────────────────────────────────────────────────────────┐
│              LAYER 2 — Feature Engineering                       │
│  Lag features | Rolling rainfall | Moving averages | Cyclical    │
└──────┬───────────────────────────────────────────────────────────┘
       │
┌──────▼───────────────────────────────────────────────────────────┐
│              LAYER 1 — Data Layer                                │
│  Inflow | Upstream Outflow | Rainfall | Temperature | Reservoir  │
│  Merge → Clean → Validate → 3,646 records (2015–2024)            │
└──────────────────────────────────────────────────────────────────┘
```

---

## 📁 Project Structure

```
HydroGPT/
│
├── 📂 data/                              # Raw and processed datasets
│   ├── inflow_data.csv                   # Daily reservoir inflow (cumecs)
│   ├── upstream_outflow.csv              # Upstream dam discharge data
│   ├── rainfall_data.csv                 # Daily catchment rainfall (mm)
│   ├── temperature_data.csv              # Daily temperature (°C)
│   ├── reservoir_level.csv               # Daily reservoir water level (m)
│   ├── merged_cleaned_dataset.csv        # Final processed dataset (auto-generated)
│   ├── train.csv                         # Training split 2015–2022
│   ├── val.csv                           # Validation split 2023
│   └── test.csv                          # Test split 2024
│
├── 📂 models/                            # Saved trained models and artifacts
│   ├── model_xgb.pkl                     # Trained XGBoost model
│   ├── model_lstm.keras                  # Trained LSTM model
│   ├── model_sarimax.pkl                 # Trained SARIMAX model
│   ├── scaler_xgb.pkl                    # Feature scaler for XGBoost
│   ├── features_xgb.pkl                  # Feature names list
│   ├── metrics_xgb.pkl                   # XGBoost RMSE/MAE/R² metrics
│   ├── metrics_lstm.pkl                  # LSTM RMSE/MAE/R² metrics
│   ├── metrics_sarimax.pkl               # SARIMAX RMSE/MAE/R² metrics
│   ├── actual_vs_predicted.png           # XGBoost prediction plot
│   ├── lstm_actual_vs_predicted.png      # LSTM prediction plot
│   ├── sarimax_actual_vs_predicted.png   # SARIMAX prediction plot
│   ├── rmse_over_iterations.png          # XGBoost training curve
│   └── feature_importance.png            # XGBoost feature importance chart
│
├── 📂 src/                               # Main source code
│   │
│   ├── 📂 data_layer/
│   │   ├── data_cleaning.py              # Merge, clean, feature engineering
│   │   └── data_validation.py            # IQR outlier check, ADF stationarity test
│   │
│   ├── 📂 algorithm_engine/
│   │   ├── train_xgboost.py              # XGBoost model training
│   │   ├── train_lstm.py                 # LSTM model training
│   │   ├── train_sarimax.py              # SARIMAX model training
│   │   └── predict.py                    # Inference for all 3 models
│   │
│   ├── 📂 agents/
│   │   ├── react_agent.py                # ReAct loop — Thought→Action→Observe→Revise
│   │   ├── tools.py                      # Forecast, risk analysis, metrics tools
│   │   ├── llm_client.py                 # Groq LLM + smart fallback
│   │   ├── model_selector.py             # Auto best-model selection by R²
│   │   └── model_registry.py             # Load and compare all model metrics
│   │
│   ├── 📂 rag/
│   │   ├── knowledge_base.txt            # Hydrology domain knowledge (19 sections)
│   │   ├── vector_store.py               # Build TF-IDF vector store
│   │   └── retriever.py                  # Cosine similarity retrieval
│   │
│   ├── 📂 api/
│   │   └── app.py                        # FastAPI application
│   │
│   └── 📂 ui/
│       └── app.py                        # Streamlit no-code dashboard
│
├── run_hydrogpt.py                       # Main launcher — starts everything
├── test_llm.py                           # Test LLM connection
├── test_rag.py                           # Test RAG retrieval
├── test_prediction.py                    # Test model inference
├── test_registry.py                      # Test model registry
├── requirements.txt                      # Python dependencies
└── README.md                             # This file
```

---

## 🛠️ Technologies Used

### Machine Learning & Deep Learning

| Library | Version | Purpose |
|---------|---------|---------|
| XGBoost | 1.7+ | Gradient boosting forecasting model |
| LSTM | via TensorFlow 2.12+ | Long Short-Term Memory deep learning model |
| SARIMAX | via Statsmodels 0.14+ | Seasonal ARIMA statistical forecasting model |
| TensorFlow / Keras | 2.12+ | LSTM deep learning model |
| Statsmodels | 0.14+ | SARIMAX statistical model |
| Scikit-learn | 1.3+ | Scalers, metrics, TF-IDF vectorizer |

### Data Processing

| Library | Version | Purpose |
|---------|---------|---------|
| Pandas | 2.0+ | Data loading, merging, manipulation |
| NumPy | 1.24+ | Numerical operations and arrays |
| Matplotlib | 3.7+ | Training plots and visualizations |
| Joblib | 1.3+ | Save and load models and scalers |

### AI / LLM / RAG

| Tool | Purpose |
|------|---------|
| Groq API | Free LLaMA 3 inference (no credit card needed) |
| LLaMA 3.1 8B Instant | Primary LLM for natural language explanations |
| TF-IDF Vectorizer | Knowledge base vectorization for RAG |
| Cosine Similarity | Semantic search for document retrieval |

### Backend & Frontend

| Library | Version | Purpose |
|---------|---------|---------|
| FastAPI | 0.104+ | REST API backend framework |
| Uvicorn | 0.24+ | ASGI server for FastAPI |
| Streamlit | 1.28+ | No-code interactive dashboard |
| Pydantic | 2.0+ | API request validation |
| Requests | 2.31+ | HTTP calls from UI to API |

---

## 📊 Dataset Description

### Raw Data Sources (5 files)

| File | Description | Unit |
|------|-------------|------|
| `inflow_data.csv` | Daily reservoir inflow — **target variable** | cumecs (m³/s) |
| `upstream_outflow.csv` | Upstream dam discharge | cumecs |
| `rainfall_data.csv` | Daily catchment rainfall | mm |
| `temperature_data.csv` | Daily air temperature | °C |
| `reservoir_level.csv` | Daily reservoir water level | meters |

### Processed Dataset Summary

```
Total Records  : 3,646 daily entries
Time Span      : January 2015 — December 2024  (10 years)
Total Columns  : 21  (5 raw + 14 engineered + date index)
Target Variable: inflow_cumecs
Missing Values : 0  (after preprocessing)
```

### Train / Validation / Test Split

| Split | Period | Records | Purpose |
|-------|--------|---------|---------|
| Training | 2015–2022 | 2,915 | Model learning |
| Validation | 2023 | 365 | Hyperparameter tuning |
| Test | 2024 | 366 | Final evaluation |

> ⚠️ **Strictly temporal split — zero data leakage guaranteed.**

---

## ⚙️ Feature Engineering

14 meaningful features created from raw data to capture hydrological patterns:

### Inflow Lag Features — River Memory Effect
```python
inflow_lag1 = inflow shifted by 1 day    # Yesterday's inflow
inflow_lag3 = inflow shifted by 3 days   # 3-day memory effect
inflow_lag7 = inflow shifted by 7 days   # Weekly flow pattern
```

### Rainfall Lag Features — Delayed Runoff Effect
```python
rain_lag1   = rainfall shifted by 1 day  # Immediate runoff delay
rain_lag3   = rainfall shifted by 3 days # Catchment response delay
```

### Rolling Rainfall — Soil Saturation Indicator
```python
rain_3day   = sum of last 3 days rainfall  # Short-term accumulation
rain_5day   = sum of last 5 days rainfall  # Medium-term saturation
```

### Moving Averages — Trend Smoothing
```python
inflow_ma3  = 3-day rolling mean of inflow  # Short-term trend
inflow_ma7  = 7-day rolling mean of inflow  # Weekly trend
```

### Cyclical Date Encoding — Seasonal Awareness
```python
month_sin, month_cos = sin/cos(2π × month / 12)      # Monthly cycle
day_sin,   day_cos   = sin/cos(2π × dayofyear / 365)  # Annual cycle
```

> 💡 **Why cyclical encoding?** Month numbers (1–12) make December and January seem far apart but they are adjacent winter months. Sin/cos encoding makes all months continuous and circular so the model understands seasonality correctly.

---

## 🤖 Forecasting Models

### 1. XGBoost — Primary Model

Builds 500 decision trees sequentially, each correcting errors of the previous.

```python
XGBRegressor(
    n_estimators     = 500,    # Number of boosting rounds
    learning_rate    = 0.03,   # Slow learning = better generalization
    max_depth        = 6,      # Tree complexity control
    subsample        = 0.8,    # 80% samples per tree
    colsample_bytree = 0.8,    # 80% features per tree
    eval_metric      = "rmse"
)
```
**Preprocessing:** StandardScaler (zero mean, unit variance)

---

### 2. LSTM — Deep Learning Model

Learns long-term temporal dependencies via memory gates.

```
Input   → (14 days × 13 features)
LSTM    → 128 units, return_sequences=True
Dropout → 0.3
LSTM    → 64 units
Dropout → 0.3
Dense   → 32 units (ReLU)
Output  → 1 unit (predicted inflow)
```
**Settings:** Adam optimizer · MSE loss · 14-day lookback · EarlyStopping(patience=10) · MinMaxScaler

---

### 3. SARIMAX — Statistical Baseline

Captures seasonal patterns, trends, and external variable effects.

```python
SARIMAX(
    order          = (2, 1, 2),     # AR=2, Differencing=1, MA=2
    seasonal_order = (1, 1, 1, 7),  # Weekly seasonal cycle
)
```
**Components:** AR (past inflow) · I (trend removal) · MA (error correction) · Seasonal (monsoon cycle) · X (rainfall, temperature)

---

### Automatic Model Selection

The system automatically compares RMSE of all trained models and selects the best one at runtime — no manual decision needed.

---

## 📚 RAG Knowledge System

HydroGPT uses Retrieval-Augmented Generation (Lewis et al., 2020) to answer hydrology questions accurately without hallucination.

### RAG Pipeline

```
STEP 1 — BUILD (one time)
knowledge_base.txt  →  Split into chunks  →  TF-IDF vectorize  →  vector_store.pkl

STEP 2 — RETRIEVE (every query)
User Question  →  TF-IDF transform  →  Cosine similarity search  →  Top 3 chunks

STEP 3 — AUGMENT
Retrieved chunks + User question  →  Structured LLM prompt

STEP 4 — GENERATE
LLM prompt  →  Groq LLaMA 3 / Smart Fallback  →  Grounded accurate answer
```

### Knowledge Base — 19 Sections
Hydrology fundamentals · Rainfall-runoff relationships · Reservoir inflow dynamics · Hydropower generation · Reservoir operations · Flood risk assessment · XGBoost / LSTM / SARIMAX explanations · Feature engineering · RAG and ReAct frameworks · FastAPI and Streamlit · Data preprocessing · Model evaluation · India water resources · LLMs and NLP · HydroGPT results

---

## 🔄 ReAct Intelligent Agent

Based on Yao et al. (2022) — the agent interleaves reasoning with tool execution in a loop.

### Agent Loop (up to 3 iterations)

```
User Query: "Predict inflow for January 2024"
      │
      ▼
ITERATION 1
  Thought     : "I need to run the inflow forecast first"
  Action      : forecast_tool(start_date, end_date)
  Observation : { predicted inflow values for each day }
      │
      ▼
ITERATION 2
  Thought     : "Now assess flood risk based on predictions"
  Action      : risk_analysis_tool(start_date, end_date)
  Observation : { average_inflow: 241.5, risk_level: "LOW RISK" }
      │
      ▼
ITERATION 3
  Thought     : "Retrieve hydrology knowledge to enrich explanation"
  Action      : retrieve_context(query)
  Observation : { relevant hydrology knowledge chunks }
      │
      ▼
FINAL ANSWER
  All observations  →  LLM  →  Comprehensive natural language explanation
```

### Available Tools

| Tool | Function | Returns |
|------|----------|---------|
| `forecast_tool()` | Runs best ML model inference | Predicted inflow series |
| `risk_analysis_tool()` | Classifies flood risk | LOW / MODERATE / HIGH |
| `model_metrics_tool()` | Loads all model metrics | RMSE, MAE, R² per model |
| `retrieve_context()` | RAG knowledge retrieval | Top 3 knowledge chunks |

### Flood Risk Classification

| Average Inflow | Risk Level | Action Required |
|----------------|------------|-----------------|
| < 300 cumecs | ✅ LOW RISK | Normal operations |
| 300–500 cumecs | ⚠️ MODERATE RISK | Enhanced monitoring |
| > 500 cumecs | 🚨 HIGH FLOOD RISK | Emergency protocol |

---

## 🔌 API Endpoints

Base URL: `http://127.0.0.1:8000`

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | System health check |
| POST | `/predict` | Full ReAct forecast pipeline |
| POST | `/chat` | RAG-powered hydrology Q&A |
| GET | `/best-model` | Best model name + R² score |
| GET | `/model-metrics` | All model RMSE / MAE / R² |

**Example — Run Forecast:**
```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"start_date": "2024-01-01", "end_date": "2024-01-07"}'
```

**Example — Ask a Question:**
```bash
curl -X POST http://127.0.0.1:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "How does rainfall affect reservoir inflow?"}'
```

> 📖 Interactive API documentation available at `http://127.0.0.1:8000/docs`

---

## 🖥️ No-Code Dashboard

The Streamlit dashboard provides a complete no-code interface with 3 tabs:

**Tab 1 — 📈 Inflow Forecast**
- Date range pickers and model selection
- Run Forecast button triggers the full ReAct agent pipeline
- 4 metric cards: Avg Inflow · Risk Level · Model Used · R²
- Color-coded flood alert banner (Green / Orange / Red)
- Interactive forecast line chart with CSV download
- AI explanation panel with LLM-generated summary
- Expandable ReAct reasoning trace for transparency

**Tab 2 — 💬 HydroGPT Assistant**
- 6 suggested question quick-access buttons
- Free-text question input
- RAG-powered answer with expandable knowledge context

**Tab 3 — 📊 Model Performance**
- RMSE / MAE / R² comparison table for all 3 models
- Bar chart visualization
- Best model highlighted automatically

---

## 🔧 Installation and Setup

### Prerequisites
- Python 3.10 or higher
- pip package manager
- 4GB+ RAM (8GB recommended for LSTM training)

### Step 1 — Clone the Repository
```bash
git clone https://github.com/nadargeAnkita/HydroGPT.git
cd HydroGPT
```

### Step 2 — Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 3 — Get Free Groq API Key
1. Go to **https://console.groq.com**
2. Sign up with your Google account — **no credit card needed**
3. Go to **API Keys** → **Create API Key**
4. Copy your key (starts with `gsk_...`)

### Step 4 — Add API Key to Project

Open `run_hydrogpt.py` and add:
```python
os.environ["GROQ_API_KEY"] = "gsk_your_actual_key_here"
```

> **Without a key:** HydroGPT works fully using the smart fallback mode covering 15+ hydrology topics. Add the key anytime to enable full LLaMA 3 responses.

---

## 🚀 How to Run

### First Time — Run Training Scripts

```bash
# Step 1: Build and clean the merged dataset
python -m src.data_layer.data_cleaning

# Step 2: Validate data quality
python -m src.data_layer.data_validation

# Step 3: Train all models
python -m src.algorithm_engine.train_xgboost
python -m src.algorithm_engine.train_lstm
python -m src.algorithm_engine.train_sarimax

# Step 4: Build RAG vector store
python -m src.rag.vector_store
```

### Launch the Full System

```bash
python run_hydrogpt.py
```

| Service | URL |
|---------|-----|
| 🎨 Streamlit Dashboard | http://localhost:8501 |
| ⚡ FastAPI Backend | http://127.0.0.1:8000 |
| 📖 API Documentation | http://127.0.0.1:8000/docs |

### Run Individual Tests

```bash
python test_llm.py        # Test LLM connection
python test_rag.py        # Test RAG retrieval
python test_prediction.py # Test model inference
python test_registry.py   # Test model registry
```

---

## 📈 Model Performance Results

### Phase 1 — XGBoost on 2024 Test Set

| Metric | Value | Interpretation |
|--------|-------|----------------|
| **R²** | **0.862** | Model explains 86.2% of inflow variation |
| **RMSE** | **15.593 cumecs** | Average prediction error ≈ ±15 cumecs |
| **MAE** | **12.461 cumecs** | Typical daily error ≈ 12 cumecs |

### Key Observations
- **Stable training:** Validation RMSE decreased steadily over 500 iterations — no overfitting
- **Seasonal accuracy:** Model correctly follows monsoon peaks and dry season recessions
- **Top features:** `inflow_lag1`, `inflow_lag3`, `rain_3day` are the most important predictors
- **Zero data leakage:** Strict temporal split ensures genuinely unseen test evaluation

---

## 📖 References

1. **Li et al. (2025)** — D2: An LLM agent for visual AI modeling. https://doi.org/10.1016/j.egyai.2025.100582
2. **Ren et al. (2024)** — WaterGPT: LLM for Hydrology. https://www.mdpi.com/2073-4441/16/21/3075
3. **Yao et al. (2022)** — ReAct: Synergizing Reasoning and Acting in LLMs. https://arxiv.org/abs/2210.03629
4. **Lewis et al. (2020)** — Retrieval-Augmented Generation. https://arxiv.org/abs/2005.11401
5. **Vaswani et al. (2017)** — Attention Is All You Need. https://arxiv.org/abs/1706.03762
6. **Chen & Guestrin (2016)** — XGBoost: A Scalable Tree Boosting System. https://dl.acm.org/doi/10.1145/2939672.2939785
7. **Hochreiter & Schmidhuber (1997)** — Long Short-Term Memory. https://www.bioinf.jku.at/publications/older/2604.pdf
8. **Shneiderman (2020)** — Human-Centered AI. https://www.nature.com/articles/s42256-020-00235-9
9. **Kisi & Parmar (2016)** — Hydrological time series forecasting using LSTM. https://www.sciencedirect.com/science/article/pii/S0022169416305699
10. **India WRIS Portal** — https://indiawris.gov.in

---

## 👩‍💻 Author

- **Ankita Nadarge**
- **Vaidehi Mangrule**
- **Pranali Oulkar**
- **Sakshi Jadhav**
- Final Year Major Project · B.Tech. Computer Engineering and Technology 
- Under the Guidance of - Dr. Chetan J. Awati 

---

## 📄 License

This project is developed for academic purposes as a Final Year Major Project.

---

<p align="center">
  <b>HydroGPT</b> · AI-Driven Hydropower Forecasting · ML + RAG + ReAct Agent<br>
</p>
