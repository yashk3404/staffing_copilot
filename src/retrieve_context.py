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


def build_project_summary(project: dict) -> str:
    """
    Item 22 -- same descriptive-sentence shape build_score_matrix's
    offline pipeline bakes into project_profiles.json for premade
    projects ("Looking for <roles>. Required skills: <skills>.
    Minimum <N> years experience. Deadline in <N> days. <priority>
    priority <budget> budget project."), built on the fly here
    instead -- a custom (C0xx) project created mid-session has no
    project_profiles.json entry, that file is generated offline and
    predates Phase 6 entirely.

    Mirrors matcher._build_profile_text() in spirit: read whatever
    fields are actually on the record, skip anything missing rather
    than print "None" or crash, since save_project() doesn't enforce
    every field being present.
    """
    roles = [r.strip() for r in str(project.get("required_roles", "") or "")
             .split(";") if r.strip()]
    skills = [s.strip() for s in str(project.get("required_skills", "") or "")
              .split(";") if s.strip()]

    parts = []
    if roles:
        parts.append(f"Looking for {' and '.join(roles)}.")
    if skills:
        parts.append(f"Required skills: {', '.join(skills)}.")

    min_exp = project.get("min_experience")
    if min_exp not in (None, ""):
        parts.append(f"Minimum {min_exp} years experience.")

    deadline = project.get("deadline_days")
    if deadline not in (None, ""):
        parts.append(f"Deadline in {deadline} days.")

    priority = project.get("priority")
    budget   = project.get("budget_band")
    if priority or budget:
        parts.append(
            f"{priority or 'Unspecified'} priority "
            f"{budget or 'unspecified'} budget project."
        )

    return " ".join(parts) if parts else \
        f"{project.get('project_name', 'Custom project')} -- no further details provided."


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

    def retrieve_adhoc(self, project_id: str, role: str,
                        project: dict, candidate_pool: dict) -> dict:
        """
        Item 22 -- ad-hoc equivalent of retrieve(), for custom (C0xx)
        projects. These never get a score_matrix.csv row, a
        staffing_plan.csv row, or a project_profiles.json /
        employee_profiles.json entry -- all four are generated
        offline against the premade 30-project set only, before this
        project ever existed. Everything retrieve() reads from those
        files, this rebuilds from what's already in memory instead:

        project:        the custom project record from
                         project_store.get_project_by_id() -- needs
                         "assignments" (role -> employee_id) already
                         filled in by update_project_assignments().
        candidate_pool: {role: DataFrame} from
                         project_store.get_candidate_pool(project_id)
                         -- the pool item 21 started saving, same
                         shape match_all_roles_adhoc() returns
                         (employee_id, name, role, experience_years,
                         availability_pct, skills, final_score,
                         eligible, ...). Every field retrieve() would
                         otherwise look up in self.employees is
                         already sitting in this DataFrame, since
                         match_adhoc() reads it straight off
                         employees_df at solve time.

        Returns the exact same shape retrieve() does (project_id,
        role, project, assigned, runner_up, n_eligible) so
        generate_explanation.build_prompt() and dashboard.py's
        runner-up panel work against either one unmodified. Returns
        {"error": ...} instead if the pool is missing (project never
        solved, or solved in a session predating item 21) or the
        role/assignment isn't in it -- callers should show that
        message rather than assume every custom project has this.
        """
        if not candidate_pool:
            return {"error": f"No candidate pool saved for {project_id} "
                              f"-- was it solved after item 21 landed?"}

        assigned_emp_id = project.get("assignments", {}).get(role)
        if not assigned_emp_id:
            return {"error": f"No assignment found for {project_id} / {role}"}

        role_df = candidate_pool.get(role)
        if role_df is None or role_df.empty:
            return {"error": f"No saved candidates for role {role} "
                              f"on {project_id}"}

        assigned_rows = role_df[role_df.employee_id == assigned_emp_id]
        if assigned_rows.empty:
            return {"error": f"Assigned employee {assigned_emp_id} not "
                              f"found in saved candidate pool for "
                              f"{project_id} / {role}"}

        assigned_row   = assigned_rows.iloc[0]
        assigned_score = float(assigned_row["final_score"])

        assigned_info = {
            "employee_id":      assigned_emp_id,
            "name":             assigned_row.get("name",             assigned_emp_id),
            "actual_role":      assigned_row.get("role",             ""),
            "experience_years": assigned_row.get("experience_years", ""),
            "availability_pct": assigned_row.get("availability_pct", ""),
            "skills":           assigned_row.get("skills",           ""),
            "score":            round(assigned_score, 4),
            "profile":          self._build_employee_profile_text(assigned_row),
        }

        project_info = {
            "project_id": project_id,
            "name":       project.get("project_name", project_id),
            "client":     project.get("client",        ""),
            "deadline":   project.get("deadline_days", ""),
            "skills":     project.get("required_skills", ""),
            "summary":    build_project_summary(project),
        }

        # Runner-up: best eligible candidate who wasn't assigned.
        # candidate_pool DataFrames come straight out of
        # match_all_roles_adhoc(), already sorted by final_score
        # descending -- re-sort anyway rather than trust that
        # ordering survived a session_state round-trip.
        eligible = role_df[role_df["eligible"] == True]  # noqa: E712
        runner_up_rows = (
            eligible[eligible.employee_id != assigned_emp_id]
            .sort_values("final_score", ascending=False)
        )

        runner_up = None
        if not runner_up_rows.empty:
            ru_row   = runner_up_rows.iloc[0]
            ru_score = float(ru_row["final_score"])
            runner_up = {
                "employee_id":      ru_row["employee_id"],
                "name":             ru_row.get("name",             ru_row["employee_id"]),
                "actual_role":      ru_row.get("role",             ""),
                "experience_years": ru_row.get("experience_years", ""),
                "availability_pct": ru_row.get("availability_pct", ""),
                "skills":           ru_row.get("skills",           ""),
                "score":            round(ru_score, 4),
                "score_gap":        round(ru_score - assigned_score, 4),
            }

        return {
            "project_id": project_id,
            "role":       role,
            "project":    project_info,
            "assigned":   assigned_info,
            "runner_up":  runner_up,
            "n_eligible": int(len(eligible)),
        }

    @staticmethod
    def _build_employee_profile_text(row: pd.Series) -> str:
        """
        Same descriptive-sentence shape as
        matcher.Matcher._build_profile_text() (and the offline
        embed_employees.py pipeline that produced
        employee_profiles.json) -- built here on the fly for
        candidate_pool rows, since neither a real employee scored via
        match_adhoc() nor a custom (CE0xx) one has an
        employee_profiles.json entry to read a pre-baked profile
        string from. Deliberately no Beginner/Intermediate/Expert
        grouping, matching matcher's own note that the flat skill
        list this app collects has no per-skill level to group by.
        """
        skills = [s.strip() for s in str(row.get("skills", "") or "")
                  .split(";") if s.strip()]
        parts = [
            f"{row.get('role', '')} with "
            f"{row.get('experience_years', 0)} years of experience."
        ]
        if skills:
            parts.append(f"Skilled in {', '.join(skills)}.")
        parts.append(f"Available at {row.get('availability_pct', 0)}% capacity.")
        return " ".join(parts)

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