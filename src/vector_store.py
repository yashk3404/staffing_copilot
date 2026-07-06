# src/vector_store.py
"""
Builds and queries a FAISS vector store for fast candidate retrieval.

Run:
    python src/vector_store.py
"""

import faiss
import numpy as np
import pandas as pd
import json
from pathlib import Path


class VectorStore:
    """
    FAISS-backed vector store for employee embeddings.
    Supports fast top-k nearest-neighbour search.
    """

    def __init__(self, embedding_dim: int = 384):
        self.dim        = embedding_dim
        self.index      = None
        self.employee_df = None
        self.profiles   = []

    # ── Build ─────────────────────────────────────────────────────

    def build(self, embeddings: np.ndarray,
              employee_df: pd.DataFrame,
              profiles: list) -> None:
        """Load embeddings into FAISS index."""
        emb = embeddings.astype("float32")

        # Inner product index (works with normalized vectors = cosine sim)
        self.index = faiss.IndexFlatIP(self.dim)
        self.index.add(emb)

        self.employee_df = employee_df.reset_index(drop=True)
        self.profiles    = profiles
        print(f"  ✅ FAISS index built: {self.index.ntotal} vectors")

    # ── Save / Load ───────────────────────────────────────────────

    def save(self, out_dir: str) -> None:
        out = Path(out_dir)
        faiss.write_index(self.index, str(out / "employee_faiss.index"))
        print(f"  ✅ FAISS index saved → employee_faiss.index")

    def load(self, data_dir: str) -> None:
        d = Path(data_dir)
        self.index       = faiss.read_index(str(d / "employee_faiss.index"))
        self.employee_df = pd.read_csv(d / "employees_with_index.csv",
                                       index_col=0)
        with open(d / "employee_profiles.json") as f:
            raw = json.load(f)
        self.profiles = [r["profile"] for r in raw]
        print(f"  ✅ Loaded FAISS index: {self.index.ntotal} vectors")

    # ── Search ────────────────────────────────────────────────────

    def search(self, query_vector: np.ndarray, top_k: int = 5) -> pd.DataFrame:
        """
        Find top-k most similar employees for a query vector.
        Returns DataFrame with employee info + similarity score.
        """
        q = query_vector.astype("float32").reshape(1, -1)

        # Normalize query (cosine similarity via inner product)
        faiss.normalize_L2(q)

        scores, indices = self.index.search(q, top_k)

        results = []
        for rank, (idx, score) in enumerate(zip(indices[0], scores[0]), 1):
            if idx == -1:
                continue
            row = self.employee_df.iloc[idx].to_dict()
            row["similarity_score"] = round(float(score), 4)
            row["rank"]             = rank
            row["profile_text"]     = self.profiles[idx]
            results.append(row)

        return pd.DataFrame(results)

    def search_by_text(self, query_text: str,
                       model,
                       top_k: int = 5) -> pd.DataFrame:
        """Encode a text query then search."""
        vec = model.encode([query_text], normalize_embeddings=True)
        return self.search(vec[0], top_k=top_k)


# ── Main ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    from sentence_transformers import SentenceTransformer

    BASE = Path(__file__).parent.parent / "data" / "processed"

    print("\n── Building Vector Store ──────────────────────────")

    # Load pre-computed embeddings
    embeddings   = np.load(BASE / "employee_embeddings.npy")
    employee_df  = pd.read_csv(BASE / "employees_with_index.csv", index_col=0)
    with open(BASE / "employee_profiles.json") as f:
        profiles = [r["profile"] for r in json.load(f)]

    # Build + save index
    store = VectorStore(embedding_dim=embeddings.shape[1])
    store.build(embeddings, employee_df, profiles)
    store.save(str(BASE))

    # Test: search with a sample query
    print("\n── Test Searches ──────────────────────────────────")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    queries = [
        "Python developer with machine learning and data pipelines",
        "Senior Android developer with Kotlin and REST APIs",
        "DevOps engineer with Kubernetes and AWS",
    ]

    for q in queries:
        print(f"\nQuery: '{q}'")
        print("-" * 60)
        results = store.search_by_text(q, model, top_k=5)
        for _, row in results.iterrows():
            print(f"  #{int(row['rank'])} [{row['similarity_score']:.4f}]"
                  f"  {row['name']:<18} | {row['role']:<18}"
                  f" | {row['experience_years']}yr"
                  f" | {row['availability_pct']}% avail")

    print("\n── Done ───────────────────────────────────────────\n")