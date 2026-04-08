from src.agents.llm_client import call_llm

messages = [
    {"role": "system", "content": "You are a hydrology forecasting expert."},
    {"role": "user", "content": "Explain how rainfall affects reservoir inflow."}
]

response = call_llm(messages)

print(response)