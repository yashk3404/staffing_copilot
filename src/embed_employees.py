# src/embed_employees.py
"""
Encodes every employee's skill profile into a semantic vector.

Run:
    python src/embed_employees.py
"""

from sentence_transformers import SentenceTransformer
import numpy as np
import pandas as pd
import json
from pathlib import Path


class EmployeeEmbedder:
    """Converts employee skill profiles into embedding vectors."""

    MODEL_NAME = "all-MiniLM-L6-v2"

    def __init__(self):
        print(f"  Loading model: {self.MODEL_NAME}")
        self.model      = SentenceTransformer(self.MODEL_NAME)
        self.df         = None
        self.embeddings = None
        self.profiles   = []

    # ── Data loading ─────────────────────────────────────────────

    def load(self, csv_path: str) -> None:
        self.df = pd.read_csv(csv_path)
        print(f"  Loaded {len(self.df)} employees")

    # ── Profile text builder ──────────────────────────────────────

    def _build_profile(self, row: pd.Series) -> str:
        """
        Convert one employee row into a rich descriptive sentence.
        This is what gets semantically embedded.

        Example:
            "Senior ML Engineer with 7 years experience.
             Expert in Python, TensorFlow, PyTorch.
             Intermediate in pandas, NumPy.
             Available at 60% capacity."
        """
        skills       = [s.strip() for s in row["skills"].split(";")]
        proficiency  = [p.strip() for p in row["proficiency"].split(";")]

        # Group skills by proficiency
        expert       = [s for s, p in zip(skills, proficiency) if p == "Expert"]
        intermediate = [s for s, p in zip(skills, proficiency) if p == "Intermediate"]
        beginner     = [s for s, p in zip(skills, proficiency) if p == "Beginner"]

        parts = [
            f"{row['role']} with {row['experience_years']} years of experience.",
        ]
        if expert:
            parts.append(f"Expert in {', '.join(expert)}.")
        if intermediate:
            parts.append(f"Intermediate in {', '.join(intermediate)}.")
        if beginner:
            parts.append(f"Beginner in {', '.join(beginner)}.")
        parts.append(f"Available at {row['availability_pct']}% capacity.")

        return " ".join(parts)

    # ── Embedding ─────────────────────────────────────────────────

    def embed(self) -> None:
        print("  Building profile texts...")
        self.profiles   = [self._build_profile(row)
                           for _, row in self.df.iterrows()]
        print("  Encoding embeddings...")
        self.embeddings = self.model.encode(
            self.profiles, show_progress_bar=True,
            batch_size=32, normalize_embeddings=True
        )
        print(f"  ✅ Embeddings shape: {self.embeddings.shape}")

    # ── Save ──────────────────────────────────────────────────────

    def save(self, out_dir: str) -> None:
        out = Path(out_dir)

        # 1. numpy array
        np.save(out / "employee_embeddings.npy", self.embeddings)

        # 2. profile texts (for debugging / RAG retrieval)
        with open(out / "employee_profiles.json", "w") as f:
            json.dump(
                [{"employee_id": row["employee_id"],
                  "name":        row["name"],
                  "profile":     profile}
                 for (_, row), profile in zip(self.df.iterrows(), self.profiles)],
                f, indent=2
            )

        # 3. metadata (for display in dashboard)
        self.df.to_csv(out / "employees_with_index.csv", index=True)

        print(f"  ✅ Saved to {out}/")
        print(f"     employee_embeddings.npy")
        print(f"     employee_profiles.json")
        print(f"     employees_with_index.csv")

    def get_dataframe(self) -> pd.DataFrame:
        return self.df.copy()

    def get_embeddings(self) -> np.ndarray:
        return self.embeddings.copy()

    def get_profiles(self) -> list:
        return self.profiles.copy()


# ── Main ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    BASE = Path(__file__).parent.parent / "data" / "processed"

    print("\n── Employee Embedder ──────────────────────────────")
    embedder = EmployeeEmbedder()
    embedder.load(str(BASE / "employees.csv"))
    embedder.embed()
    embedder.save(str(BASE))
    print("── Done ───────────────────────────────────────────\n")