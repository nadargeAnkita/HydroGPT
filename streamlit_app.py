"""
HydroGPT — Streamlit Cloud Entry Point
Runs standalone without FastAPI (direct function calls).
Main module: streamlit_app.py
"""

import os
import sys
import streamlit as st
import pandas as pd

# ─────────────────────────────────────────
# Step 1 — Add project root to path
# ─────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ─────────────────────────────────────────
# Step 2 — Load Groq API key FIRST
# Must happen before any src imports
# ─────────────────────────────────────────
def _load_groq_key():
    # From Streamlit secrets (Streamlit Cloud)
    try:
        if hasattr(st, "secrets") and "GROQ_API_KEY" in st.secrets:
            key = str(st.secrets["GROQ_API_KEY"]).strip()
            if key:
                os.environ["GROQ_API_KEY"] = key
                return key
    except Exception:
        pass

    # From environment variable
    key = os.environ.get("GROQ_API_KEY", "").strip()
    if key:
        return key

    # From .env file
    try:
        env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
        if os.path.exists(env_path):
            with open(env_path) as f:
                for line in f:
                    if line.strip().startswith("GROQ_API_KEY"):
                        key = line.strip().split("=", 1)[1].strip().strip('"').strip("'")
                        if key:
                            os.environ["GROQ_API_KEY"] = key
                            return key
    except Exception:
        pass

    return None

groq_key = _load_groq_key()

# ─────────────────────────────────────────
# Step 3 — Page Config
# ─────────────────────────────────────────
st.set_page_config(
    page_title="HydroGPT",
    page_icon="💧",
    layout="wide"
)

st.markdown("""
<style>
    .main-header { font-size:2rem; font-weight:700; color:#1a73e8; }
    .sub-header  { font-size:1rem; color:#666; margin-bottom:1.5rem; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">💧 HydroGPT</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">AI-Driven Hydropower Inflow Forecasting · '
    'ML + RAG + ReAct Agent</div>',
    unsafe_allow_html=True
)

# ─────────────────────────────────────────
# Step 4 — System Status
# ─────────────────────────────────────────
st.subheader("🔧 System Status")
c1, c2, c3, c4 = st.columns(4)

models_ok = os.path.exists(os.path.join("models", "model_xgb.pkl"))
rag_ok    = os.path.exists(os.path.join("src", "rag", "vector_store.pkl"))
llm_ok    = bool(groq_key)

with c1: st.success("✅ App Running")
with c2: st.success("✅ ML Models") if models_ok else st.warning("⚠️ Models not found")
with c3: st.success("✅ Knowledge Base") if rag_ok else st.warning("⚠️ Vector store missing")
with c4: st.success("✅ LLM (Groq)") if llm_ok else st.warning("⚠️ LLM Fallback Mode")

# ─────────────────────────────────────────
# Step 5 — Tabs
# ─────────────────────────────────────────
tab1, tab2, tab3 = st.tabs([
    "📈 Inflow Forecast",
    "💬 HydroGPT Assistant",
    "📊 Model Performance"
])

# ═══════════════════════════════════════
# TAB 1 — INFLOW FORECAST
# ═══════════════════════════════════════
with tab1:
    st.subheader("Reservoir Inflow Forecast")
    col_left, col_right = st.columns([1, 2])

    with col_left:
        st.markdown("**Forecast Settings**")
        start_date = st.date_input("📅 Start Date", value=pd.to_datetime("2024-01-01"))
        end_date   = st.date_input("📅 End Date",   value=pd.to_datetime("2024-01-31"))
        run_btn    = st.button("🚀 Run Forecast", use_container_width=True)

    with col_right:
        if run_btn:
            if end_date <= start_date:
                st.error("End date must be after start date.")
            else:
                with st.spinner("Running HydroGPT ReAct Agent..."):
                    try:
                        from src.agents.react_agent import run_forecast_agent

                        result = run_forecast_agent(
                            start_date=str(start_date),
                            end_date=str(end_date)
                        )

                        forecast    = result.get("forecast", {}).get("forecast", {})
                        model_used  = result.get("forecast", {}).get("model_used", "xgboost")
                        model_r2    = result.get("forecast", {}).get("model_r2", 0)
                        risk_level  = result.get("risk_analysis", {}).get("risk_level", "N/A")
                        avg_inflow  = result.get("risk_analysis", {}).get("average_inflow", 0)
                        explanation = result.get("llm_explanation", "")

                        # Metrics
                        m1, m2, m3, m4 = st.columns(4)
                        m1.metric("Avg Inflow (cumecs)", f"{avg_inflow:.1f}")
                        m2.metric("Risk Level", risk_level)
                        m3.metric("Model Used", model_used.upper())
                        m4.metric("Model R²", f"{model_r2:.3f}")

                        # Flood alert banner
                        if "HIGH" in str(risk_level):
                            st.error(f"🚨 {risk_level} — Immediate action required!")
                        elif "MODERATE" in str(risk_level):
                            st.warning(f"⚠️ {risk_level} — Monitor closely.")
                        else:
                            st.success(f"✅ {risk_level} — Normal operations.")

                        # Forecast chart
                        if forecast:
                            st.subheader("Inflow Forecast Chart")
                            df_fc = pd.DataFrame(
                                list(forecast.items()),
                                columns=["date", "Inflow (cumecs)"]
                            )
                            df_fc["date"] = pd.to_datetime(df_fc["date"])
                            df_fc = df_fc.set_index("date")
                            st.line_chart(df_fc)

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

                        # ReAct reasoning trace
                        with st.expander("🔍 View ReAct Agent Reasoning Trace"):
                            for step in result.get("reasoning_trace", []):
                                st.markdown(
                                    f"**[Iteration {step.get('iteration', '')}]"
                                    f" {step.get('step', '')}**"
                                )
                                st.text(str(step.get("content", ""))[:300])
                                st.divider()

                    except FileNotFoundError as e:
                        st.error(f"❌ Model file not found: {e}\nPlease train models first.")
                    except Exception as e:
                        st.error(f"❌ Forecast failed: {str(e)}")
        else:
            st.info("👈 Set forecast dates and click **Run Forecast** to get started.")

# ═══════════════════════════════════════
# TAB 2 — HYDROGPT CHAT ASSISTANT
# ═══════════════════════════════════════
with tab2:
    st.subheader("💬 Ask HydroGPT")
    st.markdown(
        "Ask anything about hydrology, inflow forecasting, "
        "flood risk, or reservoir operations."
    )

    # Suggested questions
    suggestions = [
        "What are lag features in hydrology?",
        "How does rainfall affect inflow?",
        "What causes high flood risk?",
        "How does LSTM forecast inflow?",
        "What is the monsoon effect on reservoirs?",
        "Explain XGBoost vs LSTM for forecasting"
    ]

    st.markdown("**Suggested questions:**")
    q_cols = st.columns(3)
    for i, q in enumerate(suggestions):
        with q_cols[i % 3]:
            if st.button(q, key=f"sq_{i}", use_container_width=True):
                st.session_state["chat_input"] = q

    st.divider()

    user_question = st.text_input(
        "Your question:",
        value=st.session_state.get("chat_input", ""),
        placeholder="e.g. What are lag features in hydrology?"
    )

    if st.button("Ask HydroGPT 💬", use_container_width=True):
        if user_question.strip():
            with st.spinner("HydroGPT is thinking..."):
                try:
                    from src.rag.retriever import retrieve_context
                    from src.agents.llm_client import call_llm

                    # Get RAG context
                    context = retrieve_context(user_question, top_k=3)

                    # Build messages for LLM
                    messages = [
                        {
                            "role": "system",
                            "content": (
                                "You are HydroGPT, an expert in hydrology and hydropower "
                                "inflow forecasting. Answer the question clearly, "
                                "in detail, and in a structured format using the context provided. "
                                "Use bullet points and sections where appropriate."
                            )
                        },
                        {
                            "role": "user",
                            "content": (
                                f"Context from HydroGPT Knowledge Base:\n{context}\n\n"
                                f"Question: {user_question}"
                            )
                        }
                    ]

                    answer = call_llm(messages)
                    st.success(answer)

                    with st.expander("📚 Knowledge Retrieved from RAG"):
                        st.text(context)

                except Exception as e:
                    st.error(f"❌ Assistant error: {str(e)}")
        else:
            st.warning("Please enter a question first.")

# ═══════════════════════════════════════
# TAB 3 — MODEL PERFORMANCE
# ═══════════════════════════════════════
with tab3:
    st.subheader("📊 Model Performance Comparison")

    try:
        from src.agents.model_registry import load_metrics

        metrics = load_metrics()

        if metrics:
            rows = []
            for model_name, m in metrics.items():
                rows.append({
                    "Model": model_name.upper(),
                    "RMSE":  round(m.get("RMSE", 0), 3),
                    "MAE":   round(m.get("MAE",  0), 3),
                    "R²":    round(m.get("R2",   0), 3)
                })

            df_m = pd.DataFrame(rows)
            st.dataframe(df_m, use_container_width=True)

            # Bar chart using plotly to avoid typing_extensions issue
            try:
                import plotly.express as px
                fig = px.bar(
                    df_m,
                    x="Model",
                    y=["RMSE", "MAE"],
                    barmode="group",
                    title="Model Performance Comparison"
                )
                st.plotly_chart(fig, use_container_width=True)
            except Exception:
                st.bar_chart(df_m.set_index("Model")[["RMSE", "MAE"]])

            best = min(
                metrics,
                key=lambda m: metrics[m].get("RMSE", float("inf"))
            )
            st.success(f"🏆 Best Model: **{best.upper()}** (lowest RMSE)")

        else:
            st.info("No trained model metrics found.")
            st.markdown("**Phase 1 XGBoost Results (from report):**")
            st.table(pd.DataFrame([{
                "Model": "XGBoost",
                "RMSE": 15.593,
                "MAE": 12.461,
                "R²": 0.862
            }]))

    except Exception as e:
        st.warning(f"Could not load metrics: {e}")
        st.markdown("**Phase 1 XGBoost Results (from report):**")
        st.table(pd.DataFrame([{
            "Model": "XGBoost",
            "RMSE": 15.593,
            "MAE": 12.461,
            "R²": 0.862
        }]))

# ─────────────────────────────────────────
# Footer
# ─────────────────────────────────────────
st.divider()
st.caption(
    "HydroGPT · AI-Driven Hydropower Forecasting · "
    "ML + RAG + ReAct Agent | Final Year Project"
)
