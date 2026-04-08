# 💧 HydroGPT — AI-Driven Hydropower Inflow Forecasting Platform

HydroGPT is a domain-specific intelligent agent for hydropower reservoir inflow forecasting. It combines **XGBoost**, **LSTM**, and **SARIMAX** forecasting models with a **RAG knowledge base** and a **ReAct reasoning agent** — all accessible through a no-code **Streamlit interface**.

---

## 🏗️ Project Structure

```
HydroGPT/
├── data/                         # Hydrological datasets
│   ├── inflow_data.csv
│   ├── upstream_outflow.csv
│   ├── rainfall_data.csv
│   ├── temperature_data.csv
│   ├── reservoir_level.csv
│   └── merged_cleaned_dataset.csv
│
├── models/                       # Saved trained models & metrics
│   ├── model_xgb.pkl
│   ├── model_lstm.keras
│   ├── model_sarimax.pkl
│   ├── metrics_xgb.pkl
│   ├── metrics_lstm.pkl
│   └── metrics_sarimax.pkl
│
├── src/
│   ├── data_layer/
│   │   ├── data_cleaning.py      # Merge, clean, feature engineering
│   │   └── data_validation.py    # IQR outlier check, ADF stationarity test
│   │
│   ├── algorithm_engine/
│   │   ├── train_xgboost.py      # XGBoost training
│   │   ├── train_lstm.py         # LSTM training
│   │   ├── train_sarimax.py      # SARIMAX training
│   │   └── predict.py            # Inference for all models
│   │
│   ├── agents/
│   │   ├── react_agent.py        # ReAct loop: Thought→Action→Observe→Revise
│   │   ├── tools.py              # Forecast, risk analysis, metrics tools
│   │   ├── llm_client.py         # Groq LLM client + smart fallback
│   │   ├── model_selector.py     # Auto-select best model by R²
│   │   └── model_registry.py     # Load & compare all model metrics
│   │
│   ├── rag/
│   │   ├── knowledge_base.txt    # Hydrology domain knowledge
│   │   ├── vector_store.py       # Build TF-IDF / embedding vector store
│   │   └── retriever.py          # Semantic retrieval for RAG
│   │
│   ├── api/
│   │   └── app.py                # FastAPI backend
│   │
│   └── ui/
│       └── app.py                # Streamlit no-code dashboard
│
├── run_hydrogpt.py               # Launch everything at once
├── requirements.txt
└── README.md
```

---

## ⚙️ Setup Instructions

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Set Groq API Key (Free — no credit card needed)

Get your free key at **https://console.groq.com**

**Windows:**
```cmd
set GROQ_API_KEY=gsk_your_key_here
```

**Mac/Linux:**
```bash
export GROQ_API_KEY=gsk_your_key_here
```

> **Without a key:** HydroGPT still works using a smart rule-based fallback. Add the key anytime to enable full LLM responses.

### 3. Run Training (first time only)

```bash
# Step 1: Build the dataset
python -m src.data_layer.data_cleaning

# Step 2: Validate data
python -m src.data_layer.data_validation

# Step 3: Train models
python -m src.algorithm_engine.train_xgboost
python -m src.algorithm_engine.train_lstm
python -m src.algorithm_engine.train_sarimax

# Step 4: Build RAG vector store
python -m src.rag.vector_store
```

### 4. Launch HydroGPT

```bash
python run_hydrogpt.py
```

This starts:
- **FastAPI API**: http://127.0.0.1:8000
- **Streamlit Dashboard**: http://localhost:8501

---

## 🚀 Usage

### No-Code Dashboard (Streamlit)

1. Open **http://localhost:8501**
2. Go to **Inflow Forecast** tab → Set dates → Click **Run Forecast**
3. View forecast chart, flood risk level, and AI explanation
4. Go to **HydroGPT Assistant** tab → Ask any hydrology question
5. Go to **Model Performance** tab → Compare XGBoost / LSTM / SARIMAX

### API (FastAPI)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Health check |
| `/predict` | POST | Run forecast + ReAct agent |
| `/chat` | POST | Ask hydrology question (RAG + LLM) |
| `/best-model` | GET | Get best model name + R² |
| `/model-metrics` | GET | All model metrics |

**Example API call:**
```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"start_date": "2024-01-01", "end_date": "2024-01-07"}'
```

---

## 📊 Model Performance (Phase 1 Results)

| Model | RMSE | MAE | R² |
|-------|------|-----|----|
| XGBoost | 15.593 | 12.461 | 0.862 |
| LSTM | — | — | — |
| SARIMAX | — | — | — |

---

## 🧠 System Architecture

```
User Query (Natural Language)
        ↓
  Streamlit UI (No-Code Interface)
        ↓
  FastAPI Backend (/predict, /chat)
        ↓
  ReAct Agent Loop
  ┌─────────────────────────────┐
  │ Thought → Action → Observe  │
  │      → Revise → Repeat      │
  └─────────────────────────────┘
        ↓                ↓
  ML Engine           RAG Layer
  (XGBoost/LSTM/    (Vector DB +
   SARIMAX)          Knowledge Base)
        ↓                ↓
       LLM Explanation (Groq / Fallback)
        ↓
  Final Answer to User
```

---

## 📚 References

See Phase 1 report for full references including Li et al. (2025), Yao et al. (2022), Lewis et al. (2020), Vaswani et al. (2017), Chen & Guestrin (2016), and Hochreiter & Schmidhuber (1997).

---

## 👩‍💻 Author

Ankita Nadarge — Final Year Major Project
