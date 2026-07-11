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
        self.unstaffable_by_exclusion = []  # populated by build()

    # ── Build ─────────────────────────────────────────────────────

    def build(self, projects_to_staff: list = None, exclude_ids: set = None):
        """
        Build decision variables and constraints.
        projects_to_staff: list of project_ids to include, or None for all.
        exclude_ids: item 24 -- employee_ids to drop from the candidate
                     pool entirely before building anything, e.g.
                     everyone already assigned in an earlier Simulate-
                     mode batch (see dashboard.py's sim_busy_ids). Kept
                     as a plain set-membership filter on self.df, same
                     shape as match_adhoc()'s exclude_ids over in
                     matcher.py, rather than a new constraint on top of
                     the model -- excluding a row here means it never
                     gets a decision variable at all, so it can't be
                     assigned no matter what the objective would
                     prefer. Everything downstream (decision variables,
                     both constraints, the objective) already just
                     iterates self.df, so nothing else in this class
                     needs to change for this to work.
        """
        df = self.scores[self.scores["eligible"] == True].copy()
        if projects_to_staff is not None:
            df = df[df["project_id"].isin(projects_to_staff)]

        # Slots that had at least one eligible candidate BEFORE
        # exclude_ids is applied -- these are the role-slots this
        # build is actually on the hook to staff. Captured before the
        # exclude_ids filter below so a slot that gets wiped out
        # entirely by exclusion is still remembered here.
        slots_before_exclusion = set(
            df[["project_id", "role"]].drop_duplicates().itertuples(index=False, name=None)
        )

        if exclude_ids:
            df = df[~df["employee_id"].isin(exclude_ids)]

        self.df = df.reset_index(drop=True)

        # Any slot that existed pre-exclusion but has zero rows left
        # post-exclusion would otherwise just be absent from every
        # groupby below -- no decision variables, no "== 1" constraint,
        # nothing stopping the solver from returning OPTIMAL having
        # silently never staffed it. That's worse than a crash: a
        # caller checking `result.empty` sees False and has no signal
        # anything went wrong. Force the whole model INFEASIBLE
        # instead via a direct contradiction, one per wiped-out slot,
        # so solve() reports it the same honest way a genuinely
        # unsolvable model does.
        slots_after_exclusion = set(
            self.df[["project_id", "role"]].drop_duplicates().itertuples(index=False, name=None)
        )
        self.unstaffable_by_exclusion = sorted(
            slots_before_exclusion - slots_after_exclusion
        )
        for pid, role in self.unstaffable_by_exclusion:
            marker = self.model.NewBoolVar(f"unstaffable_{pid}_{role}")
            self.model.Add(marker == 1)
            self.model.Add(marker == 0)

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
            if self.unstaffable_by_exclusion:
                # Item 24 -- distinguish "genuinely no eligible pool
                # left, plus overlapping-role contention" from
                # ambiguous INFEASIBLE. self.unstaffable_by_exclusion
                # (set in build()) already pinpoints exactly which
                # (project, role) slots exclude_ids wiped out --
                # surface that here instead of a generic "try
                # relaxing constraints" message that doesn't tell the
                # caller anything actionable when the real cause is
                # known precisely.
                slots = ", ".join(f"{pid}/{role}"
                                   for pid, role in self.unstaffable_by_exclusion)
                print(f"   Specifically unstaffable due to exclude_ids "
                      f"(zero eligible candidates left): {slots}")
            else:
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

        # Item 24 follow-up -- results can legitimately be empty even
        # on a non-INFEASIBLE status: an empty projects_to_staff list,
        # or every requested project/role already fully excluded
        # (self.x has zero entries -- an edge case build() itself
        # can't always turn into INFEASIBLE, since "zero variables"
        # isn't the same situation as "a slot existed but got wiped
        # out," which is what the unstaffable_by_exclusion markers
        # cover). pd.DataFrame([]).sort_values(["project_id", "role"])
        # raises KeyError (no columns to sort by on an empty frame
        # built from an empty list) -- build the empty frame with the
        # right columns explicitly instead of letting that crash.
        if results:
            result_df = pd.DataFrame(results).sort_values(["project_id", "role"])
        else:
            result_df = pd.DataFrame(
                columns=["project_id", "role", "employee_id", "final_score"]
            )
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

# ── Ad-hoc project wiring (Phase 3 item 11) ─────────────────────────

def staff_custom_project(project: dict,
                          matcher,
                          staffing_plan_df: pd.DataFrame,
                          employees_df: pd.DataFrame = None,
                          min_avail: int = 60,
                          time_limit_sec: int = 10,
                          verbose: bool = True) -> pd.DataFrame:
    """
    The full pipeline for one already-saved custom project (Phase 4's
    "Create Project" form, after save_project() has assigned it an
    id): exclude everyone already busy -> score every role's
    candidates -> solve -> write the result back into storage.

    project:          dict with a project_id already assigned (i.e.
                       the return value of project_store.save_project(),
                       looked back up) -- required, since
                       update_project_assignments() needs it at the
                       end and there's nowhere else to get it from.
    matcher:           a loaded Matcher instance (matcher.load()
                       already called) -- passed in rather than
                       constructed here, since loading it (the
                       sentence-transformer model + embeddings) is
                       expensive and the dashboard should only do it
                       once per session, not once per project.
    staffing_plan_df:  the premade staffing_plan.csv, read by the
                       caller -- same "only one place reads the real
                       path" convention as project_store.py.
    employees_df:      item 17 -- pass a merged pool (e.g.
                       employee_store.load_all_employees(employees))
                       so session-added custom employees are actually
                       eligible for this solve, not just the premade
                       roster. Defaults to None, which makes
                       match_all_roles_adhoc() fall back to the
                       matcher's own premade-only self.emp_df --
                       existing callers are unaffected.

    Imports project_store functions locally (not at module level) to
    avoid a hard import-time dependency between optimize_staffing.py
    (a plain script, importable outside Streamlit) and project_store.py
    (which needs a live Streamlit runtime for st.session_state) --
    this function is the only place in optimize_staffing.py that
    actually needs Streamlit-backed state, so it's the only place
    that pays for the import.

    Returns the same DataFrame solve_ad_hoc_project() would (role,
    employee_id, final_score) -- empty if any role had zero eligible
    candidates or the solve was infeasible. On the empty-result path,
    storage is deliberately left untouched (no partial/empty
    assignments written) so a failed attempt doesn't silently
    overwrite a project's previous good result.
    """
    from src.project_store import (
        get_busy_employee_ids,
        update_project_assignments,
        save_candidate_pool,
    )

    busy = get_busy_employee_ids(staffing_plan_df)

    role_scores = matcher.match_all_roles_adhoc(
        project=project,
        exclude_ids=busy,
        min_avail=min_avail,
        verbose=verbose,
        employees_df=employees_df,
    )

    # Item 21 -- persist the full candidate pool (not just the final
    # assignment) so runner-up data and LLM explanations survive past
    # this call. Saved even on a later infeasible/empty solve -- some
    # roles here may still have real scored candidates worth keeping
    # for an explanation of what went wrong.
    save_candidate_pool(project["project_id"], role_scores)

    result_df = solve_ad_hoc_project(role_scores, time_limit_sec=time_limit_sec)

    if result_df.empty:
        if verbose:
            print(f"   Not staffed -- leaving project "
                  f"{project.get('project_id')} without assignments.")
        return result_df

    assignments = dict(zip(result_df["role"], result_df["employee_id"]))
    update_project_assignments(project["project_id"], assignments)

    if verbose:
        print(f"   Project {project['project_id']} staffed and saved: "
              f"{assignments}")

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