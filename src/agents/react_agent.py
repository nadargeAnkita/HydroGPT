"""
ReAct Agent — Reasoning + Acting loop for HydroGPT.

Flow per report (Chapter 4.5):
  Thought → Action → Observation → Revise → Repeat → Final Answer
"""

from src.agents.tools import forecast_tool, risk_analysis_tool, model_metrics_tool
from src.rag.retriever import retrieve_context
from src.agents.llm_client import call_llm


MAX_ITERATIONS = 3


def run_forecast_agent(start_date: str, end_date: str):
    """
    Full ReAct loop:
    1. Thought  — reason about what needs to be done
    2. Action   — call tools (forecast, risk, RAG)
    3. Observe  — collect tool results
    4. Revise   — check if answer is complete
    5. Respond  — generate final LLM explanation
    """

    reasoning_trace = []
    context = {}
    iteration = 0

    print("\n=== HydroGPT ReAct Agent Starting ===")

    while iteration < MAX_ITERATIONS:
        iteration += 1

        # -----------------------------------------------
        # STEP 1: THOUGHT — what does the agent need?
        # -----------------------------------------------
        thought = _think(iteration, context)
        reasoning_trace.append({"step": "Thought", "iteration": iteration, "content": thought})
        print(f"\n[Iteration {iteration}] Thought: {thought}")

        # -----------------------------------------------
        # STEP 2: ACTION — call the right tool
        # -----------------------------------------------
        action, action_input = _decide_action(iteration, context)
        reasoning_trace.append({"step": "Action", "iteration": iteration, "content": action})
        print(f"[Iteration {iteration}] Action: {action}")

        # -----------------------------------------------
        # STEP 3: OBSERVATION — execute tool, get result
        # -----------------------------------------------
        observation = _execute_action(action, action_input, start_date, end_date)
        context[action] = observation
        reasoning_trace.append({"step": "Observation", "iteration": iteration, "content": str(observation)[:200]})
        print(f"[Iteration {iteration}] Observation: {str(observation)[:100]}...")

        # -----------------------------------------------
        # STEP 4: REVISE — is the answer complete?
        # -----------------------------------------------
        if _is_complete(context):
            print(f"[Iteration {iteration}] Agent has sufficient information. Generating final answer.")
            break

    # -----------------------------------------------
    # STEP 5: FINAL ANSWER — LLM generates explanation
    # -----------------------------------------------
    final_answer = _generate_final_answer(context, reasoning_trace)
    print("\n=== ReAct Agent Complete ===\n")

    return {
        "forecast": context.get("forecast_tool", {}),
        "risk_analysis": context.get("risk_analysis_tool", {}),
        "model_metrics": context.get("model_metrics_tool", {}),
        "llm_explanation": final_answer,
        "reasoning_trace": reasoning_trace,
        "iterations": iteration
    }


def _think(iteration, context):
    """Generate a thought based on what has been collected so far."""
    if iteration == 1:
        return "I need to run the inflow forecast first to get predicted values for the requested period."
    elif iteration == 2:
        if "forecast_tool" not in context:
            return "Forecast failed. I should try again or retrieve knowledge to provide guidance."
        return "Forecast complete. Now I need to assess flood risk based on the predicted inflow values."
    elif iteration == 3:
        return "Risk assessed. I should retrieve domain knowledge to enrich the final explanation."
    return "Reviewing results to generate the final comprehensive answer."


def _decide_action(iteration, context):
    """Decide which tool to call based on iteration and context state."""
    if "forecast_tool" not in context:
        return "forecast_tool", None
    elif "risk_analysis_tool" not in context:
        return "risk_analysis_tool", None
    elif "rag_context" not in context:
        return "rag_context", None
    else:
        return "model_metrics_tool", None


def _execute_action(action, action_input, start_date, end_date):
    """Execute the chosen action/tool and return the observation."""
    try:
        if action == "forecast_tool":
            return forecast_tool(start_date, end_date)

        elif action == "risk_analysis_tool":
            return risk_analysis_tool(start_date, end_date)

        elif action == "rag_context":
            query = "Explain reservoir inflow forecasting, flood risk, and hydropower operations."
            return retrieve_context(query)

        elif action == "model_metrics_tool":
            return model_metrics_tool()

        else:
            return {"error": f"Unknown action: {action}"}

    except Exception as e:
        return {"error": str(e)}


def _is_complete(context):
    """Check if agent has enough information to generate a final answer."""
    required = ["forecast_tool", "risk_analysis_tool"]
    return all(k in context for k in required)


def _generate_final_answer(context, reasoning_trace):
    """Use LLM to generate a natural language explanation from all collected context."""

    forecast_data = context.get("forecast_tool", {})
    risk_data = context.get("risk_analysis_tool", {})
    knowledge = context.get("rag_context", "No additional knowledge retrieved.")
    metrics = context.get("model_metrics_tool", {})

    # Build prompt
    messages = [
        {
            "role": "system",
            "content": (
                "You are HydroGPT, an expert hydrology and hydropower forecasting assistant. "
                "Analyze the forecast and risk data and explain it clearly to a hydropower engineer. "
                "Keep your response practical, concise, and actionable."
            )
        },
        {
            "role": "user",
            "content": f"""
Forecast Data:
{forecast_data}

Risk Analysis:
{risk_data}

Model Performance Metrics:
{metrics}

Hydrology Knowledge:
{knowledge}

Based on the above, provide:
1. A clear summary of the inflow forecast
2. The flood risk level and what it means
3. Recommended actions for reservoir operators
4. Key factors driving the forecast
"""
        }
    ]

    return call_llm(messages)
