"""
RAG Retriever — retrieves relevant hydrology knowledge for a given query.
Supports both TF-IDF and sentence-transformer vector stores.
"""

import os
import joblib
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
VECTOR_PATH = os.path.join(BASE_DIR, "src", "rag", "vector_store.pkl")


def retrieve_context(query: str, top_k: int = 3) -> str:
    """
    Retrieve the top_k most relevant knowledge chunks for a query.

    Args:
        query: The user's question or search string
        top_k: Number of chunks to retrieve

    Returns:
        String of concatenated relevant knowledge chunks
    """

    if not os.path.exists(VECTOR_PATH):
        # Auto-build if missing
        print("Vector store not found. Building now...")
        from src.rag.vector_store import build_vector_store
        build_vector_store()

    store = joblib.load(VECTOR_PATH)
    method = store.get("method", "tfidf")
    documents = store["documents"]

    if method == "sentence_transformers":
        results = _retrieve_with_embeddings(query, store, documents, top_k)
    else:
        results = _retrieve_with_tfidf(query, store, documents, top_k)

    return "\n\n".join(results)


def _retrieve_with_tfidf(query, store, documents, top_k):
    """Retrieve using TF-IDF cosine similarity."""
    vectorizer = store["vectorizer"]
    vectors = store["vectors"]

    query_vec = vectorizer.transform([query])
    similarity = cosine_similarity(query_vec, vectors).flatten()

    top_indices = similarity.argsort()[-top_k:][::-1]
    return [documents[i] for i in top_indices if similarity[i] > 0]


def _retrieve_with_embeddings(query, store, documents, top_k):
    """Retrieve using sentence-transformer embeddings."""
    from sentence_transformers import SentenceTransformer

    vectors = store["vectors"]
    model_name = store.get("model", "all-MiniLM-L6-v2")

    embedder = SentenceTransformer(model_name)
    query_vec = embedder.encode([query])

    similarity = cosine_similarity(query_vec, vectors).flatten()
    top_indices = similarity.argsort()[-top_k:][::-1]
    return [documents[i] for i in top_indices]
