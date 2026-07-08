# src/optimize_staffing.py
"""
OR-Tools CP-SAT optimizer: turns score_matrix.csv into a conflict-free
staffing plan (no employee double-booked beyond their availability).

Run:
    python src/optimize_staffing.py
"""

import pandas as pd
from pathlib import Path
from ortools.sat.python import cp_model


class StaffingOptimizer:
    def __init__(self, score_matrix_path: str, employees_path: str):
        self.scores = pd.read_csv(score_matrix_path)
        self.employees = pd.read_csv(employees_path, index_col=0)
        self.model = cp_model.CpModel()
        self.x = {}
        self.solver = cp_model.CpSolver()
        self.df = None

    # ── Build ─────────────────────────────────────────────────────

    def build(self, projects_to_staff: list = None):
        """
        Build decision variables and constraints.
        projects_to_staff: list of project_ids to include, or None for all.
        """
        df = self.scores[self.scores["eligible"] == True].copy()
        if projects_to_staff is not None:
            df = df[df["project_id"].isin(projects_to_staff)]

        self.df = df.reset_index(drop=True)

        # Decision variables: one bool per eligible (project, role, employee)
        for _, row in self.df.iterrows():
            key = (row["employee_id"], row["project_id"], row["role"])
            self.x[key] = self.model.NewBoolVar(f"x_{key[0]}_{key[1]}_{key[2]}")

        # Constraint 1: exactly one person per (project, role)
        for (pid, role), group in self.df.groupby(["project_id", "role"]):
            vars_for_slot = [self.x[(r["employee_id"], pid, role)]
                              for _, r in group.iterrows()]
            self.model.Add(sum(vars_for_slot) == 1)

        # Constraint 2: each employee assigned to at most ONE role total
        for emp_id, group in self.df.groupby("employee_id"):
            vars_for_emp = [self.x[(emp_id, r["project_id"], r["role"])]
                             for _, r in group.iterrows()]
            self.model.Add(sum(vars_for_emp) <= 1)

        # Objective: maximize total match score (scaled to int for CP-SAT)
        objective_terms = []
        for _, row in self.df.iterrows():
            key = (row["employee_id"], row["project_id"], row["role"])
            score_int = int(round(row["final_score"] * 10000))
            objective_terms.append(score_int * self.x[key])

        self.model.Maximize(sum(objective_terms))
        print(f"   Built model: {len(self.x)} variables, "
              f"{self.df['project_id'].nunique()} projects, "
              f"{self.df.groupby(['project_id','role']).ngroups} role-slots")

    # ── Solve ─────────────────────────────────────────────────────

    def solve(self, time_limit_sec: int = 30) -> pd.DataFrame:
        self.solver.parameters.max_time_in_seconds = time_limit_sec
        status = self.solver.Solve(self.model)
        status_name = self.solver.StatusName(status)
        print(f"  Solver status: {status_name}")

        if status == cp_model.INFEASIBLE:
            print("   Model is INFEASIBLE — constraints cannot all be satisfied.")
            print("  Try: relaxing constraints, or reducing project count.")
            return pd.DataFrame()

        if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            print("   No feasible solution found.")
            return pd.DataFrame()

        results = []
        for key, var in self.x.items():
            if self.solver.Value(var):
                emp_id, pid, role = key
                row = self.df[(self.df.employee_id == emp_id) &
                               (self.df.project_id == pid) &
                               (self.df.role == role)].iloc[0]
                results.append({
                    "project_id": pid,
                    "role": role,
                    "employee_id": emp_id,
                    "final_score": row["final_score"],
                })

        result_df = pd.DataFrame(results).sort_values(["project_id", "role"])
        print(f"   Assigned {len(result_df)} role-slots | "
              f"Total score: {self.solver.ObjectiveValue() / 10000:.2f}")
        return result_df

    # ── Diagnostics ───────────────────────────────────────────────

    def find_unstaffable_roles(self, projects_to_staff: list = None) -> pd.DataFrame:
        """Roles with zero eligible candidates BEFORE building the model."""
        df = self.scores.copy()
        if projects_to_staff is not None:
            df = df[df["project_id"].isin(projects_to_staff)]

        eligible_counts = (df[df.eligible == True]
                            .groupby(["project_id", "role"]).size()
                            .reset_index(name="n_eligible"))
        all_roles = df[["project_id", "role"]].drop_duplicates()
        merged = all_roles.merge(eligible_counts, on=["project_id", "role"], how="left")
        merged["n_eligible"] = merged["n_eligible"].fillna(0).astype(int)

        return merged[merged.n_eligible == 0]


def solve_ad_hoc_project(role_scores: dict,
                          time_limit_sec: int = 10) -> pd.DataFrame:
    """
    Solve the assignment problem for ONE not-yet-saved project, given
    {role: DataFrame} from Matcher.match_all_roles_adhoc().

    This is a separate, lightweight function rather than a method on
    StaffingOptimizer because that class is built around reading a
    full multi-project score matrix from disk (score_matrix.csv). This
    solves a single project from in-memory scores instead, and never
    touches or reshuffles any other project's assignments — busy
    employees should already be excluded upstream via match_adhoc()'s
    exclude_ids, so this only decides among the remaining pool.

    Constraints: exactly one employee per role, each employee used at
    most once within this project.

    Returns a DataFrame: role, employee_id, final_score.
    Empty DataFrame if any role has zero eligible candidates, or if
    the solve is otherwise infeasible.
    """
    model = cp_model.CpModel()
    x = {}
    rows = []

    for role, df in role_scores.items():
        eligible = df[df["eligible"] == True]
        if eligible.empty:
            print(f"   No eligible candidates for role '{role}' — "
                  f"can't staff this project as specified.")
            return pd.DataFrame()
        for _, r in eligible.iterrows():
            key = (role, r["employee_id"])
            x[key] = model.NewBoolVar(f"x_{role}_{r['employee_id']}")
            rows.append((role, r["employee_id"], r["final_score"]))

    # Exactly one employee per role
    for role in role_scores:
        vars_for_role = [x[(r, eid)] for (r, eid, _) in rows if r == role]
        model.Add(sum(vars_for_role) == 1)

    # Each employee used at most once across this project's roles
    all_employees = {eid for (_, eid, _) in rows}
    for eid in all_employees:
        vars_for_emp = [x[(role, e)] for (role, e, _) in rows if e == eid]
        if len(vars_for_emp) > 1:
            model.Add(sum(vars_for_emp) <= 1)

    objective_terms = [
        int(round(score * 10000)) * x[(role, eid)]
        for (role, eid, score) in rows
    ]
    model.Maximize(sum(objective_terms))

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit_sec
    status = solver.Solve(model)

    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        print(f"   Ad-hoc solve failed: {solver.StatusName(status)}")
        return pd.DataFrame()

    assigned = []
    for (role, eid, score) in rows:
        if solver.Value(x[(role, eid)]) == 1:
            assigned.append({
                "role":        role,
                "employee_id": eid,
                "final_score": score,
            })

    result_df = pd.DataFrame(assigned)
    print(f"   Ad-hoc solve OK — {len(result_df)} role(s) assigned | "
          f"Total score: {solver.ObjectiveValue() / 10000:.2f}")
    return result_df

# ── Main ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    BASE = Path(__file__).parent.parent / "data" / "processed"

    print("\n── Staffing Optimizer ─────────────────────────────")

    opt = StaffingOptimizer(
        score_matrix_path=str(BASE / "score_matrix.csv"),
        employees_path=str(BASE / "employees_with_index.csv"),
    )

    PROJECTS = ["P001", "P002", "P003", "P004", "P005"]   # start small, scale up Day 20

    unstaffable = opt.find_unstaffable_roles(projects_to_staff=PROJECTS)
    if len(unstaffable):
        print("\n   Unstaffable roles (zero eligible candidates):")
        print(unstaffable.to_string(index=False))

    opt.build(projects_to_staff=PROJECTS)
    plan = opt.solve(time_limit_sec=60)

    print("\n── Staffing Plan ───────────────────────────────────")
    print(plan.to_string(index=False))

    out_path = BASE / "staffing_plan.csv"
    plan.to_csv(out_path, index=False)
    print(f"\n   Saved → {out_path}")
    print("── Done ────────────────────────────────────────────\n")