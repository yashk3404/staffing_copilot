# src/matcher.py
"""
Combines semantic similarity + hard filters into a single match score.

Usage:
    python src/matcher.py
"""

import numpy as np
import pandas as pd
import json
from pathlib import Path
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


class Matcher:
    """
    Scores every employee against a project role requirement.

    Score formula:
        final_score = semantic_similarity * availability_factor * experience_factor

    Hard filters (anyone failing these gets score = 0):
        - availability_pct >= min_availability
        - experience_years >= min_experience
    """

    MODEL_NAME = "all-MiniLM-L6-v2"
    DATA_DIR   = Path(__file__).parent.parent / "data" / "processed"

    # Weights
    W_SEMANTIC     = 0.70
    W_AVAILABILITY = 0.20
    W_EXPERIENCE   = 0.10

    def __init__(self):
        print("  Loading Matcher...")
        self.model    = SentenceTransformer(self.MODEL_NAME)
        self.emp_df   = None
        self.emp_emb  = None
        self.proj_df  = None
        self.profiles = []

    # ── Load ──────────────────────────────────────────────────────

    def load(self) -> None:
        d = self.DATA_DIR

        self.emp_df  = pd.read_csv(d / "employees_with_index.csv",
                                    index_col=0)
        self.proj_df = pd.read_csv(d / "projects_with_index.csv",
                                    index_col=0)
        self.emp_emb = np.load(d / "employee_embeddings.npy")

        with open(d / "employee_profiles.json") as f:
            self.profiles = [r["profile"] for r in json.load(f)]

        print(f"   Loaded {len(self.emp_df)} employees,"
              f" {len(self.proj_df)} projects")

    # ── Score helpers ─────────────────────────────────────────────

    def _availability_factor(self, avail_pct: int,
                              min_avail: int = 60) -> float:
        if avail_pct < min_avail:
            return 0.0
        return (avail_pct - min_avail) / (100 - min_avail)

    def _experience_factor(self, years: int,
                            min_years: int,
                            max_years: int = 15) -> float:
        if years < min_years:
            return 0.0
        return min(
            0.5 + (years - min_years) / (max_years - min_years) * 0.5,
            1.0
        )

    def _skill_overlap(self, emp_skills: str,
                       req_skills: list) -> float:
        emp_set = set(s.strip().lower() for s in emp_skills.split(";"))
        req_set = set(s.strip().lower() for s in req_skills)
        if not req_set:
            return 0.0
        return len(emp_set & req_set) / len(req_set)

    # ── Core match function ───────────────────────────────────────

    def match(self,
              project_id: str,
              role:       str,
              top_k:      int  = 10,
              min_avail:  int  = 60,
              verbose:    bool = True) -> pd.DataFrame:
        """
        Score and rank all employees for a specific project role.
        Returns DataFrame with scores, sorted descending.
        """
        proj_row = self.proj_df[
            self.proj_df["project_id"] == project_id
        ]
        if proj_row.empty:
            raise ValueError(f"Project {project_id} not found")
        proj = proj_row.iloc[0]

        min_exp    = int(proj["min_experience"])
        req_skills = [s.strip()
                       for s in proj["required_skills"].split(";")]

        query_text = (
            f"{role} with skills in {', '.join(req_skills)}. "
            f"Minimum {min_exp} years experience."
        )
        query_vec = self.model.encode(
            [query_text], normalize_embeddings=True
        )

        sem_scores = cosine_similarity(query_vec, self.emp_emb)[0]

        rows = []
        for idx, emp in self.emp_df.iterrows():
            avail_f = self._availability_factor(
                emp["availability_pct"], min_avail
            )
            exp_f = self._experience_factor(
                emp["experience_years"], min_exp
            )

            if (emp["availability_pct"] < min_avail or
                    emp["experience_years"] < min_exp):
                final_score = 0.0
                eligible    = False
            else:
                sem = float(sem_scores[idx])
                final_score = (
                    self.W_SEMANTIC     * sem     +
                    self.W_AVAILABILITY * avail_f +
                    self.W_EXPERIENCE   * exp_f
                )
                eligible = True

            skill_ov = self._skill_overlap(
                emp["skills"], req_skills
            )

            rows.append({
                "employee_id":         emp["employee_id"],
                "name":                emp["name"],
                "role":                emp["role"],
                "experience_years":    emp["experience_years"],
                "availability_pct":    emp["availability_pct"],
                "cost_band":           emp["cost_band"],
                "skills":              emp["skills"],
                "semantic_score":      round(float(sem_scores[idx]), 4),
                "availability_factor": round(avail_f, 4),
                "experience_factor":   round(exp_f, 4),
                "skill_overlap":       round(skill_ov, 4),
                "final_score":         round(final_score, 4),
                "eligible":            eligible,
            })

        result_df = (
            pd.DataFrame(rows)
            .sort_values("final_score", ascending=False)
            .reset_index(drop=True)
        )
        result_df["rank"] = result_df.index + 1

        if verbose:
            print(f"\nProject: {proj['project_name']}")
            print(f"Role:    {role}  |  Min exp: {min_exp}yr"
                  f"  |  Min avail: {min_avail}%")
            print(f"Query:   '{query_text}'\n")
            top = result_df[result_df["eligible"]].head(top_k)
            print(top[["rank", "name", "role",
                        "experience_years", "availability_pct",
                        "skill_overlap", "semantic_score",
                        "final_score"]].to_string(index=False))

        return result_df

    def match_all_roles(self,
                        project_id: str,
                        top_k:      int = 5) -> dict:
        """
        Run match() for every required role in a project.
        Returns {role: DataFrame} dict.
        """
        proj_row = self.proj_df[
            self.proj_df["project_id"] == project_id
        ]
        if proj_row.empty:
            raise ValueError(f"Project {project_id} not found")

        roles = [r.strip() for r in
                  proj_row.iloc[0]["required_roles"].split(";")]
        results = {}

        print(f"\n{'='*60}")
        print(f"Matching all roles for project: {project_id}")
        print(f"Roles required: {roles}")
        print(f"{'='*60}")

        for role in roles:
            results[role] = self.match(
                project_id=project_id,
                role=role,
                top_k=top_k,
                verbose=True,
            )
        return results
    
    # ── Ad-hoc matching (for projects not yet saved to proj_df) ────

    def match_adhoc(self,
                     project:     dict,
                     role:        str,
                     exclude_ids: set  = None,
                     top_k:       int  = 10,
                     min_avail:   int  = 60,
                     verbose:     bool = True) -> pd.DataFrame:
        """
        Same scoring logic as match(), for a project that hasn't been
        saved to proj_df yet (e.g. a custom project just submitted via
        the dashboard form). Also supports excluding a set of
        employee_ids entirely — e.g. everyone already staffed
        elsewhere this session, to prevent double-booking.

        project: dict with keys "min_experience" (int) and
                 "required_skills" (";"-separated str). Optional
                 "project_name" used only for the printed header.
        role:    the specific role to score against. An ad-hoc project
                 has no single row to read every role from at once, so
                 call this once per role (see match_all_roles_adhoc
                 for the convenience wrapper).
        """
        if exclude_ids is None:
            exclude_ids = set()

        min_exp    = int(project["min_experience"])
        req_skills = [s.strip()
                       for s in project["required_skills"].split(";")]

        query_text = (
            f"{role} with skills in {', '.join(req_skills)}. "
            f"Minimum {min_exp} years experience."
        )
        query_vec = self.model.encode(
            [query_text], normalize_embeddings=True
        )
        sem_scores = cosine_similarity(query_vec, self.emp_emb)[0]

        rows = []
        for idx, emp in self.emp_df.iterrows():
            if emp["employee_id"] in exclude_ids:
                continue

            avail_f = self._availability_factor(
                emp["availability_pct"], min_avail
            )
            exp_f = self._experience_factor(
                emp["experience_years"], min_exp
            )

            if (emp["availability_pct"] < min_avail or
                    emp["experience_years"] < min_exp):
                final_score = 0.0
                eligible    = False
            else:
                sem = float(sem_scores[idx])
                final_score = (
                    self.W_SEMANTIC     * sem     +
                    self.W_AVAILABILITY * avail_f +
                    self.W_EXPERIENCE   * exp_f
                )
                eligible = True

            skill_ov = self._skill_overlap(emp["skills"], req_skills)

            rows.append({
                "employee_id":         emp["employee_id"],
                "name":                emp["name"],
                "role":                emp["role"],
                "experience_years":    emp["experience_years"],
                "availability_pct":    emp["availability_pct"],
                "cost_band":           emp["cost_band"],
                "skills":              emp["skills"],
                "semantic_score":      round(float(sem_scores[idx]), 4),
                "availability_factor": round(avail_f, 4),
                "experience_factor":   round(exp_f, 4),
                "skill_overlap":       round(skill_ov, 4),
                "final_score":         round(final_score, 4),
                "eligible":            eligible,
            })

        result_df = (
            pd.DataFrame(rows)
            .sort_values("final_score", ascending=False)
            .reset_index(drop=True)
        )
        result_df["rank"] = result_df.index + 1

        if verbose:
            print(f"\nAd-hoc project: "
                  f"{project.get('project_name', '(unsaved)')}")
            print(f"Role: {role}  |  Min exp: {min_exp}yr  |  "
                  f"Min avail: {min_avail}%  |  "
                  f"Excluded (busy): {len(exclude_ids)}")
            top = result_df[result_df["eligible"]].head(top_k)
            print(top[["rank", "name", "role", "experience_years",
                        "availability_pct", "skill_overlap",
                        "semantic_score", "final_score"]]
                  .to_string(index=False))

        return result_df

    def match_all_roles_adhoc(self,
                               project:     dict,
                               exclude_ids: set  = None,
                               top_k:       int  = 5,
                               min_avail:   int  = 60,
                               verbose:     bool = True) -> dict:
        """
        Run match_adhoc() for every role in an unsaved project's
        required_roles list. Returns {role: DataFrame}.
        """
        roles = [r.strip()
                  for r in project["required_roles"].split(";")]
        results = {}

        if verbose:
            print(f"\n{'='*60}")
            print(f"Ad-hoc matching for: "
                  f"{project.get('project_name', '(unsaved)')}")
            print(f"Roles required: {roles}")
            print(f"{'='*60}")

        for role in roles:
            results[role] = self.match_adhoc(
                project=project,
                role=role,
                exclude_ids=exclude_ids,
                top_k=top_k,
                min_avail=min_avail,
                verbose=verbose,
            )
        return results

    # ── Build full score matrix ───────────────────────────────────

    def build_score_matrix(self,
                            out_dir: Path = None) -> pd.DataFrame:
        """
        Run match() for every (project, role) combination across
        all 30 projects and save two files:

        score_matrix.csv      — minimal (project_id, role,
                                employee_id, final_score, eligible)
                                used by the optimizer.

        score_matrix_full.csv — includes feature columns
                                (semantic_score, skill_overlap,
                                availability_pct, experience_years)
                                used by SHAP in Day 28.
        """
        if out_dir is None:
            out_dir = self.DATA_DIR

        all_rows      = []
        all_rows_full = []

        projects = self.proj_df["project_id"].unique()
        print(f"\n── Building score matrix for "
              f"{len(projects)} projects ──────────")

        for project_id in projects:
            proj_row = self.proj_df[
                self.proj_df["project_id"] == project_id
            ].iloc[0]
            roles = [r.strip()
                      for r in proj_row["required_roles"].split(";")]

            for role in roles:
                df = self.match(
                    project_id=project_id,
                    role=role,
                    verbose=False,
                )

                # Minimal version (for optimizer)
                for _, row in df.iterrows():
                    all_rows.append({
                        "project_id":  project_id,
                        "role":        role,
                        "employee_id": row["employee_id"],
                        "final_score": row["final_score"],
                        "eligible":    row["eligible"],
                    })

                # Full version (for SHAP)
                for _, row in df.iterrows():
                    all_rows_full.append({
                        "project_id":       project_id,
                        "role":             role,
                        "employee_id":      row["employee_id"],
                        "semantic_score":   row["semantic_score"],
                        "skill_overlap":    row["skill_overlap"],
                        "availability_pct": row["availability_pct"],
                        "experience_years": row["experience_years"],
                        "final_score":      row["final_score"],
                        "eligible":         row["eligible"],
                    })

        score_matrix_df      = pd.DataFrame(all_rows)
        score_matrix_full_df = pd.DataFrame(all_rows_full)

        # Save both
        score_matrix_df.to_csv(
            out_dir / "score_matrix.csv", index=False
        )
        score_matrix_full_df.to_csv(
            out_dir / "score_matrix_full.csv", index=False
        )

        print(f"\n   Saved score_matrix.csv "
              f"({len(score_matrix_df)} rows)")
        print(f"   Saved score_matrix_full.csv "
              f"({len(score_matrix_full_df)} rows) "
              f"— includes SHAP feature columns")
        print(f"  Columns in full: "
              f"{list(score_matrix_full_df.columns)}")
        print(f"  ({score_matrix_df['project_id'].nunique()} projects "
              f"× {score_matrix_df['employee_id'].nunique()} employees)")

        return score_matrix_df


# ── Main ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    matcher = Matcher()
    matcher.load()

    print("\n── Single Role Match ──────────────────────────────")
    matcher.match("P001", role="Backend Dev", top_k=5)

    print("\n── All Roles Match ────────────────────────────────")
    matcher.match_all_roles("P002", top_k=5)

    print("\n── Building Full Score Matrix ─────────────────────")
    matcher.build_score_matrix()