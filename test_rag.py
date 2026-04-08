from src.rag.retriever import retrieve_context

query = "Why does rainfall increase reservoir inflow?"

context = retrieve_context(query)

print("Retrieved Knowledge:\n")
print(context)