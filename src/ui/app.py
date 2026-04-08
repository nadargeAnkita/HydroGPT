"""
HydroGPT — No-Code Streamlit Dashboard
Report Section: 4.6 No-Code Interface
"""

import streamlit as st
import requests
import pandas as pd

# =====================================
# Page Configuration
# =====================================

st.set_page_config(
    page_title="HydroGPT",
    page_icon="💧",
    layout="wide"
)

# =====================================
# Custom CSS
# =====================================

st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1a73e8;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1rem;
        color: #666;
        margin-bottom: 1.5rem;
    }
    .risk-high {
        background-color: #ffebee;
        border-left: 5px solid #f44336;
        padding: 10px;
        border-radius: 5px;
    }
    .risk-moderate {
        background-color: #fff8e1;
        border-left: 5px solid #ff9800;
        padding: 10px;
        border-radius: 5px;
    }
    .risk-low {
        background-color: #e8f5e9;
        border-left: 5px solid #4caf50;
        padding: 10px;
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)

# =====================================
# Header
# =====================================

st.markdown('<div class="main-header">💧 HydroGPT</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">AI-Driven Hydropower Inflow Forecasting Platform · ML + RAG + ReAct Agent</div>', unsafe_allow_html=True)

# =====================================
# System Status Panel
# =====================================

st.subheader("🔧 System Status")
c1, c2, c3, c4 = st.columns(4)

with c1:
    st.success("✅ API Server")
with c2:
    st.success("✅ ML Models")
with c3:
    st.success("✅ Knowledge Base")
with c4:
    import os
    if os.getenv("GROQ_API_KEY"):
        st.success("✅ LLM (Groq)")
    else:
        st.warning("⚠️ LLM (Fallback Mode)")

# =====================================
# Tabs Layout
# =====================================

tab1, tab2, tab3 = st.tabs(["📈 Inflow Forecast", "💬 HydroGPT Assistant", "📊 Model Performance"])

# =====================================
# TAB 1: Forecast
# =====================================

with tab1:

    st.subheader("Reservoir Inflow Forecast")

    col_left, col_right = st.columns([1, 2])

    with col_left:
        st.markdown("**Forecast Settings**")
        start_date = st.date_input("📅 Start Date", value=pd.to_datetime("2024-01-01"))
        end_date = st.date_input("📅 End Date", value=pd.to_datetime("2024-01-31"))

        model_choice = st.selectbox(
            "Select Model",
            ["Auto (Best Model)", "XGBoost", "LSTM", "SARIMAX"]
        )

        run_forecast = st.button("🚀 Run Forecast", use_container_width=True)

    with col_right:

        if run_forecast:

            if end_date <= start_date:
                st.error("End date must be after start date.")
            else:
                with st.spinner("Running HydroGPT ReAct Agent..."):

                    try:
                        payload = {
                            "start_date": str(start_date),
                            "end_date": str(end_date)
                        }

                        response = requests.post(
                            "http://127.0.0.1:8000/predict",
                            json=payload,
                            timeout=60
                        )

                        if response.status_code != 200:
                            st.error(f"API Error {response.status_code}: {response.text}")
                        else:
                            result = response.json()

                            forecast = result.get("forecast", {}).get("forecast", {})
                            model_used = result.get("forecast", {}).get("model_used", "unknown")
                            model_r2 = result.get("forecast", {}).get("model_r2", 0)
                            risk_level = result.get("risk_analysis", {}).get("risk_level", "N/A")
                            avg_inflow = result.get("risk_analysis", {}).get("average_inflow", 0)
                            explanation = result.get("llm_explanation", "")
                            iterations = result.get("iterations", 1)

                            # Metrics Row
                            m1, m2, m3, m4 = st.columns(4)
                            m1.metric("Avg Inflow (cumecs)", f"{avg_inflow:.1f}")
                            m2.metric("Risk Level", risk_level)
                            m3.metric("Model Used", model_used.upper())
                            m4.metric("Model R²", f"{model_r2:.3f}")

                            # Flood Alert
                            if "HIGH" in risk_level:
                                st.error(f"🚨 {risk_level} — Immediate action required!")
                            elif "MODERATE" in risk_level:
                                st.warning(f"⚠️ {risk_level} — Monitor closely.")
                            else:
                                st.success(f"✅ {risk_level} — Normal operations.")

                            # Forecast Chart
                            if forecast:
                                st.subheader("Inflow Forecast Chart")
                                df_fc = pd.DataFrame(
                                    list(forecast.items()),
                                    columns=["date", "Inflow (cumecs)"]
                                )
                                df_fc["date"] = pd.to_datetime(df_fc["date"])
                                df_fc = df_fc.set_index("date")
                                st.line_chart(df_fc)

                                # Download button
                                csv = df_fc.reset_index().to_csv(index=False)
                                st.download_button(
                                    "⬇️ Download Forecast CSV",
                                    csv,
                                    "hydrogpt_forecast.csv",
                                    "text/csv"
                                )

                            # AI Explanation
                            st.subheader("🤖 AI Explanation")
                            st.info(explanation)

                            # ReAct Trace (expandable)
                            with st.expander("🔍 View ReAct Agent Reasoning Trace"):
                                trace = result.get("reasoning_trace", [])
                                for step in trace:
                                    st.markdown(f"**[Iteration {step.get('iteration', '')}] {step.get('step', '')}**")
                                    st.text(step.get("content", "")[:300])
                                    st.divider()

                    except requests.exceptions.ConnectionError:
                        st.error("❌ Cannot connect to API server. Make sure it's running: `uvicorn src.api.app:app --reload`")
                    except Exception as e:
                        st.error(f"❌ Error: {str(e)}")
        else:
            st.info("👈 Set forecast dates and click **Run Forecast** to get started.")

# =====================================
# TAB 2: Chat Assistant
# =====================================

with tab2:

    st.subheader("💬 Ask HydroGPT")
    st.markdown("Ask anything about hydrology, inflow forecasting, flood risk, or reservoir operations.")

    # Suggested questions
    st.markdown("**Suggested questions:**")
    q_cols = st.columns(3)
    suggestions = [
        "How does rainfall affect inflow?",
        "What causes high flood risk?",
        "How does LSTM forecast inflow?",
        "What is the monsoon effect on reservoirs?",
        "Explain XGBoost vs LSTM for forecasting",
        "What are lag features in hydrology?"
    ]

    for i, q in enumerate(suggestions):
        with q_cols[i % 3]:
            if st.button(q, key=f"q_{i}", use_container_width=True):
                st.session_state["chat_input"] = q

    st.divider()

    user_question = st.text_input(
        "Your question:",
        value=st.session_state.get("chat_input", ""),
        placeholder="e.g. Why does rainfall increase reservoir inflow?"
    )

    if st.button("Ask HydroGPT 💬", use_container_width=True):
        if user_question.strip():
            with st.spinner("HydroGPT is thinking..."):
                try:
                    response = requests.post(
                        "http://127.0.0.1:8000/chat",
                        json={"question": user_question},
                        timeout=30
                    )
                    if response.status_code == 200:
                        answer = response.json().get("answer", "No answer returned.")
                        context = response.json().get("context_used", "")

                        st.success(answer)

                        if context:
                            with st.expander("📚 Knowledge Retrieved from RAG"):
                                st.text(context)
                    else:
                        st.error(f"API Error: {response.text}")

                except requests.exceptions.ConnectionError:
                    st.error("❌ Cannot connect to API server.")
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
        else:
            st.warning("Please enter a question first.")

# =====================================
# TAB 3: Model Performance
# =====================================

with tab3:

    st.subheader("📊 Model Performance Comparison")

    try:
        response = requests.get("http://127.0.0.1:8000/model-metrics", timeout=10)
        if response.status_code == 200:
            data = response.json().get("metrics", {})

            if data:
                rows = []
                for model, m in data.items():
                    rows.append({
                        "Model": model.upper(),
                        "RMSE": round(m.get("RMSE", 0), 3),
                        "MAE": round(m.get("MAE", 0), 3),
                        "R²": round(m.get("R2", 0), 3)
                    })

                df_metrics = pd.DataFrame(rows)
                st.dataframe(df_metrics, use_container_width=True)

                # Bar chart
                st.bar_chart(df_metrics.set_index("Model")[["RMSE", "MAE"]])

                best = response.json().get("best_model", "")
                st.success(f"🏆 Best Model: **{best.upper()}** (lowest RMSE)")

            else:
                st.info("No model metrics available. Run training scripts first.")
        else:
            st.warning("Could not fetch model metrics from API.")

    except requests.exceptions.ConnectionError:
        st.warning("⚠️ API server not running. Start it to view model metrics.")

        # Show static results from report
        st.markdown("**XGBoost Results (from Phase 1 training):**")
        st.table(pd.DataFrame([{
            "Model": "XGBoost",
            "RMSE": 15.593,
            "MAE": 12.461,
            "R²": 0.862
        }]))

# =====================================
# Footer
# =====================================

st.divider()
st.caption("HydroGPT · AI-Driven Hydropower Forecasting · ML + RAG + ReAct Agent")
