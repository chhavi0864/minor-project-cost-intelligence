"""
Cloud Compass – retrieval module (cloud deployment).

Uses TF-IDF + cosine similarity over the embedded knowledge base.
No FAISS, no sentence-transformers, no filesystem dependency.
"""
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from app.knowledge_base import KNOWLEDGE_CHUNKS

# ---------------------------------------------------------------------------
# Module-level lazy cache (initialised once per lambda warm start)
# ---------------------------------------------------------------------------
_vectorizer: TfidfVectorizer | None = None
_matrix = None


def _ensure_index() -> None:
    global _vectorizer, _matrix
    if _vectorizer is None:
        texts = [chunk["text"] for chunk in KNOWLEDGE_CHUNKS]
        _vectorizer = TfidfVectorizer(stop_words="english")
        _matrix = _vectorizer.fit_transform(texts)


def retrieve(query: str, top_k: int = 5) -> list[dict]:
    """
    Return up to *top_k* knowledge chunks most relevant to *query*.
    Each result is a dict with keys: id, score, text, meta.
    """
    _ensure_index()
    q_vec = _vectorizer.transform([query])
    sims = cosine_similarity(q_vec, _matrix).flatten()
    top_indices = np.argsort(sims)[::-1][:top_k]
    return [
        {
            "id": KNOWLEDGE_CHUNKS[i]["id"],
            "score": float(sims[i]),
            "text": KNOWLEDGE_CHUNKS[i]["text"],
            "meta": KNOWLEDGE_CHUNKS[i].get("meta"),
        }
        for i in top_indices
        if 0 <= i < len(KNOWLEDGE_CHUNKS)
    ]
