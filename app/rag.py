"""
Simple FAISS retrieval helper.
Functions:
  - load_index_and_meta()
  - retrieve(query, top_k=5)
"""
import os
from pathlib import Path
import json
import numpy as np

# Set OMP/MKL threading env vars BEFORE any BLAS/FAISS import to prevent DLL initialization failures
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

# Ensure offline operation so local cached weights load without network timeout
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_NAME = "all-MiniLM-L6-v2"
INDEX_PATH = BASE_DIR / "data" / "faiss" / "index.faiss"
META_PATH = BASE_DIR / "data" / "faiss" / "meta.json"

# lazy load
_index = None
_meta = None
_model = None

EMB_PATH = BASE_DIR / "data" / "faiss" / "embeddings.npy"
_embeddings = None

def _load_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer(MODEL_NAME)
    return _model

def load_index_and_meta():
    global _index, _meta, _embeddings
    if _meta is None:
        if not META_PATH.exists():
            raise FileNotFoundError("FAISS meta.json not found. Run scripts/build_faiss_index.py first.")
        with open(META_PATH, "r", encoding="utf8") as f:
            _meta = json.load(f)
    if _embeddings is None and EMB_PATH.exists():
        _embeddings = np.load(str(EMB_PATH))
    # If embeddings.npy exists, use robust numpy vector similarity directly
    # to avoid Windows Python 3.14 swigfaiss.dll crashes under pytest/multithreaded runner
    if _embeddings is not None:
        _index = None
    elif _index is None and INDEX_PATH.exists():
        try:
            import faiss
            _index = faiss.read_index(str(INDEX_PATH))
        except BaseException:
            _index = None
    return _index, _meta

def _embed_query(query):
    model = _load_model()
    emb = model.encode([query], convert_to_numpy=True)
    # normalize
    emb = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12)
    return emb.astype("float32")

def retrieve(query, top_k=5):
    """
    Returns a list of dicts: {id, score, text, meta}
    Uses FAISS if available; seamlessly falls back to exact numpy dot product.
    """
    index, meta = load_index_and_meta()
    q_emb = _embed_query(query)
    
    if index is not None:
        try:
            D, I = index.search(q_emb, top_k)
            scores = D[0].tolist()
            indices = I[0].tolist()
        except Exception:
            index = None

    if index is None:
        global _embeddings
        if _embeddings is None:
            if EMB_PATH.exists():
                _embeddings = np.load(str(EMB_PATH))
            else:
                raise FileNotFoundError("Neither FAISS index nor embeddings.npy available.")
        # Cosine similarity via inner product of normalized vectors
        sims = np.dot(_embeddings, q_emb.T).flatten()
        top_indices = np.argsort(sims)[::-1][:top_k]
        scores = [float(sims[i]) for i in top_indices]
        indices = top_indices.tolist()

    results = []
    for idx, score in zip(indices, scores):
        if idx < 0 or idx >= len(meta):
            continue
        item = meta[idx]
        results.append({
            "id": item.get("id"),
            "score": float(score),
            "text": item.get("text"),
            "meta": item.get("meta")
        })
    return results
