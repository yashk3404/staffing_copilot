# src/retrieve_context.py
"""
Retrieval layer for the RAG explanation pipeline.

For a given (project_id, role) assignment, pulls:
  - Assigned employee profile, stats, and match score
  - Project requirement summary
  - Runner-up candidate and score gap

Run:
    python src/retrieve_context.py
"""

import pandas as pd
import json
from pathlib import Path


class ContextRetriever:

    def __init__(self, data_dir: str):
        d = Path(data_dir)

        self.score_matrix = pd.read_csv(d / "score_matrix.csv")
        self.plan         = pd.read_csv(d / "staffing_plan.csv")

        # Set employee_id and project_id as index for fast .loc[] lookups
        self.employees = (
            pd.read_csv(d / "employees_with_index.csv")
            .set_index("employee_id")
        )
        self.projects = (
            pd.read_csv(d / "projects_with_index.csv")
            .set_index("project_id")
        )

        # Load employee profile texts → {employee_id: profile_text}
        with open(d / "employee_profiles.json") as f:
            raw_emp = json.load(f)
        self.emp_profiles = {r["employee_id"]: r["profile"] for r in raw_emp}

        # Load project profile texts → {project_id: profile_text}
        with open(d / "project_profiles.json") as f:
            raw_proj = json.load(f)
        self.proj_profiles = {r["project_id"]: r["profile"] for r in raw_proj}

        print(f"   ContextRetriever loaded: "
              f"{len(self.plan)} assignments, "
              f"{len(self.employees)} employees, "
              f"{len(self.projects)} projects")

    # ── Core retrieval ────────────────────────────────────────────

    def retrieve(self, project_id: str, role: str) -> dict:
        """
        Retrieve all facts needed to explain one (project, role) assignment.
        Returns a structured dict ready to be formatted into a prompt.
        """

        # 1. Find assigned employee from plan
        assigned_row = self.plan[
            (self.plan.project_id == project_id) &
            (self.plan.role == role)
        ]
        if assigned_row.empty:
            return {"error": f"No assignment found for {project_id} / {role}"}

        assigned_emp_id = assigned_row.iloc[0]["employee_id"]
        assigned_score  = float(assigned_row.iloc[0]["final_score"])

        # 2. Employee details
        emp = self.employees.loc[assigned_emp_id] \
              if assigned_emp_id in self.employees.index \
              else pd.Series()

        assigned_info = {
            "employee_id":      assigned_emp_id,
            "name":             emp.get("name",             assigned_emp_id),
            "actual_role":      emp.get("role",             ""),
            "experience_years": emp.get("experience_years", ""),
            "availability_pct": emp.get("availability_pct", ""),
            "skills":           emp.get("skills",           ""),
            "score":            round(assigned_score, 4),
            "profile":          self.emp_profiles.get(assigned_emp_id, ""),
        }

        # 3. Project details
        proj = self.projects.loc[project_id] \
               if project_id in self.projects.index \
               else pd.Series()

        project_info = {
            "project_id": project_id,
            "name":       proj.get("project_name", project_id),
            "client":     proj.get("client",        ""),
            "deadline":   proj.get("deadline_days", ""),
            "skills":     proj.get("required_skills", ""),
            "summary":    self.proj_profiles.get(project_id, ""),
        }

        # 4. Full candidate shortlist for this slot
        candidates = (
            self.score_matrix[
                (self.score_matrix.project_id == project_id) &
                (self.score_matrix.role == role) &
                (self.score_matrix.eligible == True)
            ]
            .sort_values("final_score", ascending=False)
            .reset_index(drop=True)
        )

        n_eligible = len(candidates)

        # 5. Runner-up (best candidate who wasn't assigned)
        runner_up_rows = candidates[
            candidates.employee_id != assigned_emp_id
        ]

        runner_up = None
        if not runner_up_rows.empty:
            ru_row    = runner_up_rows.iloc[0]
            ru_emp_id = ru_row["employee_id"]
            ru_score  = float(ru_row["final_score"])
            ru_emp    = self.employees.loc[ru_emp_id] \
                        if ru_emp_id in self.employees.index \
                        else pd.Series()

            runner_up = {
                "employee_id":      ru_emp_id,
                "name":             ru_emp.get("name",             ru_emp_id),
                "actual_role":      ru_emp.get("role",             ""),
                "experience_years": ru_emp.get("experience_years", ""),
                "availability_pct": ru_emp.get("availability_pct", ""),
                "skills":           ru_emp.get("skills",           ""),
                "score":            round(ru_score, 4),
                "score_gap":        round(ru_score - assigned_score, 4),
            }

        return {
            "project_id": project_id,
            "role":       role,
            "project":    project_info,
            "assigned":   assigned_info,
            "runner_up":  runner_up,
            "n_eligible": n_eligible,
        }

    def retrieve_all(self) -> list:
        """Retrieve context for every assignment in the staffing plan."""
        return [self.retrieve(row["project_id"], row["role"])
                for _, row in self.plan.iterrows()]


# ── Main ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    BASE = Path(__file__).parent.parent / "data" / "processed"

    print("\n── Context Retriever ──────────────────────────────")
    retriever = ContextRetriever(str(BASE))

    ctx = retriever.retrieve("P001", "Backend Dev")

    print("\n── Retrieved Context ───────────────────────────────")
    print(f"Project  : {ctx['project']['name']} ({ctx['project_id']})")
    print(f"Role     : {ctx['role']}")
    print(f"Assigned : {ctx['assigned']['name']} "
          f"| {ctx['assigned']['experience_years']}yr "
          f"| {ctx['assigned']['availability_pct']}% avail "
          f"| score {ctx['assigned']['score']}")
    if ctx["runner_up"]:
        ru = ctx["runner_up"]
        print(f"Runner-up: {ru['name']} "
              f"| {ru['experience_years']}yr "
              f"| {ru['availability_pct']}% avail "
              f"| score {ru['score']} "
              f"(gap: {ru['score_gap']:+.4f})")
    print(f"Pool size: {ctx['n_eligible']} eligible candidates")
    print("── Done ────────────────────────────────────────────\n")