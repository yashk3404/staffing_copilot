# src/embed_projects.py
"""
Encodes every project requirement into a semantic vector.

Run:
    python src/embed_projects.py
"""

from sentence_transformers import SentenceTransformer
import numpy as np
import pandas as pd
import json
from pathlib import Path


class ProjectEmbedder:
    """Converts project requirements into embedding vectors."""

    MODEL_NAME = "all-MiniLM-L6-v2"

    def __init__(self):
        print(f"  Loading model: {self.MODEL_NAME}")
        self.model      = SentenceTransformer(self.MODEL_NAME)
        self.df         = None
        self.embeddings = None
        self.profiles   = []

    def load(self, csv_path: str) -> None:
        self.df = pd.read_csv(csv_path)
        print(f"  Loaded {len(self.df)} projects")

    def _build_profile(self, row: pd.Series) -> str:
        """
        Convert one project row into a descriptive requirement sentence.

        Example:
            "Looking for a Backend Dev and Data Engineer.
             Required skills: Python, REST APIs, SQL, AWS.
             Minimum 3 years experience.
             Deadline in 45 days. High priority project."
        """
        roles  = [r.strip() for r in row["required_roles"].split(";")]
        skills = [s.strip() for s in row["required_skills"].split(";")]

        return (
            f"Looking for {' and '.join(roles)}. "
            f"Required skills: {', '.join(skills)}. "
            f"Minimum {row['min_experience']} years experience. "
            f"Deadline in {row['deadline_days']} days. "
            f"{row['priority'].capitalize()} priority {row['budget_band']} budget project."
        )

    def embed(self) -> None:
        print("  Building requirement texts...")
        self.profiles   = [self._build_profile(row)
                           for _, row in self.df.iterrows()]
        print("  Encoding embeddings...")
        self.embeddings = self.model.encode(
            self.profiles, show_progress_bar=True,
            batch_size=32, normalize_embeddings=True
        )
        print(f"  ✅ Embeddings shape: {self.embeddings.shape}")

    def save(self, out_dir: str) -> None:
        out = Path(out_dir)

        np.save(out / "project_embeddings.npy", self.embeddings)

        with open(out / "project_profiles.json", "w") as f:
            json.dump(
                [{"project_id":   row["project_id"],
                  "project_name": row["project_name"],
                  "profile":      profile}
                 for (_, row), profile in zip(self.df.iterrows(), self.profiles)],
                f, indent=2
            )

        self.df.to_csv(out / "projects_with_index.csv", index=True)

        print(f"  ✅ Saved to {out}/")
        print(f"     project_embeddings.npy")
        print(f"     project_profiles.json")
        print(f"     projects_with_index.csv")

    def get_dataframe(self)  -> pd.DataFrame: return self.df.copy()
    def get_embeddings(self) -> np.ndarray:   return self.embeddings.copy()
    def get_profiles(self)   -> list:         return self.profiles.copy()


if __name__ == "__main__":
    BASE = Path(__file__).parent.parent / "data" / "processed"

    print("\n── Project Embedder ───────────────────────────────")
    embedder = ProjectEmbedder()
    embedder.load(str(BASE / "projects.csv"))
    embedder.embed()
    embedder.save(str(BASE))
    print("── Done ───────────────────────────────────────────\n")