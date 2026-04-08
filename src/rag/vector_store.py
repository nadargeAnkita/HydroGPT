"""
Vector Store — builds and saves the RAG knowledge base.

Uses TF-IDF vectorization (no API key needed, works offline).
When sentence-transformers is available, uses proper embeddings instead.

Run this once to build the store:
    python -m src.rag.vector_store
"""

import os
import joblib
import numpy as np

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
KB_PATH = os.path.join(BASE_DIR, "src", "rag", "knowledge_base.txt")
VECTOR_PATH = os.path.join(BASE_DIR, "src", "rag", "vector_store.pkl")


def _chunk_knowledge_base(text, min_words=8):
    """Split KB into meaningful chunks — sentences and paragraphs."""
    chunks = []

    # Split by paragraph first
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

    for para in paragraphs:
        # Split paragraph into sentences
        sentences = [s.strip() for s in para.replace("\n", " ").split(".") if len(s.split()) >= min_words]
        chunks.extend(sentences)

    # Also add full paragraphs as chunks for broader context
    for para in paragraphs:
        if len(para.split()) >= min_words:
            chunks.append(para.replace("\n", " ").strip())

    # Deduplicate while preserving order
    seen = set()
    unique_chunks = []
    for c in chunks:
        if c not in seen:
            seen.add(c)
            unique_chunks.append(c)

    return unique_chunks


def build_vector_store():
    """Build and save the vector store from the knowledge base."""

    print("=== Building Vector Store ===")

    with open(KB_PATH, "r", encoding="utf-8") as f:
        text = f.read()

    documents = _chunk_knowledge_base(text)
    print(f"Total chunks: {len(documents)}")

    # Try sentence-transformers first (better quality)
    try:
        from sentence_transformers import SentenceTransformer
        print("Using sentence-transformers for embeddings...")

        embedder = SentenceTransformer("all-MiniLM-L6-v2")
        vectors = embedder.encode(documents, show_progress_bar=True)

        store = {
            "documents": documents,
            "vectors": vectors,
            "method": "sentence_transformers",
            "model": "all-MiniLM-L6-v2"
        }

    except ImportError:
        # Fallback: TF-IDF (works offline, no install needed)
        print("Using TF-IDF vectorization (offline mode)...")

        from sklearn.feature_extraction.text import TfidfVectorizer

        vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=5000,
            sublinear_tf=True
        )
        vectors = vectorizer.fit_transform(documents)

        store = {
            "documents": documents,
            "vectorizer": vectorizer,
            "vectors": vectors,
            "method": "tfidf"
        }

    joblib.dump(store, VECTOR_PATH)
    print(f"Vector store saved to: {VECTOR_PATH}")
    print(f"Method: {store['method']}")
    print("Vector store built successfully!")


if __name__ == "__main__":
    build_vector_store()
