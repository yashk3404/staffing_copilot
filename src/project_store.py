"""
Project storage interface -- Phase 3 item 9 (C0xx ID scheme).

Same pattern as employee_store.py: everything sits behind
st.session_state for now. When Phase 5 picks real persistence (items
19-20), only the *internals* of these functions change -- every call
site elsewhere (dashboard.py, matcher.match_adhoc(),
get_busy_employee_ids()) stays exactly the same.

Schema mirrors data/processed/projects_with_index.csv:
    project_id, project_name, client, required_roles (";"-sep),
    required_skills (";"-sep), min_experience, deadline_days,
    budget_band, priority
plus one field the real CSV doesn't have -- "assignments" -- since
custom projects don't get a staffing_plan.csv row. It's a dict of
{role: employee_id}, written by update_project_assignments() after
the mini-solver runs (item 11).
"""

import streamlit as st
import pandas as pd

from src.employee_store import load_all_employees

CUSTOM_ID_PREFIX = "C"
CUSTOM_ID_WIDTH = 3  # C001, C002, ... -- never collides with the
                     # real P0xx range from projects_with_index.csv


def _ensure_session_store():
    if "custom_projects" not in st.session_state:
        st.session_state.custom_projects = {}      # {project_id: dict}
    if "custom_project_counter" not in st.session_state:
        st.session_state.custom_project_counter = 0
    if "custom_candidate_pools" not in st.session_state:
        st.session_state.custom_candidate_pools = {}  # {project_id: {role: DataFrame}}


def _next_custom_project_id() -> str:
    _ensure_session_store()
    st.session_state.custom_project_counter += 1
    n = st.session_state.custom_project_counter
    return f"{CUSTOM_ID_PREFIX}{n:0{CUSTOM_ID_WIDTH}d}"


def save_project(record: dict) -> str:
    """
    Commits a project record -- from the Phase 4 "Create Project"
    form, post-review-edit -- into storage.

    Assigns a C0xx id if the record doesn't already have one; never
    overwrites an existing id, so calling this twice on the same
    record (e.g. after update_project_assignments()) updates it in
    place instead of creating a duplicate.

    record is expected to have at least: project_name, client,
    required_roles (";"-separated str), required_skills
    (";"-separated str), min_experience, deadline_days, budget_band,
    priority -- same shape match_adhoc()/match_all_roles_adhoc()
    already expect. Extra keys are stored as-is; this function
    doesn't validate shape, that's the form's job.

    "assignments" defaults to {} if not present -- gets filled in
    by update_project_assignments() once the mini-solver runs.

    Returns the assigned (or existing) project_id.
    """
    _ensure_session_store()
    proj_id = record.get("project_id")
    if not proj_id:
        proj_id = _next_custom_project_id()
        record = {**record, "project_id": proj_id}
    record.setdefault("assignments", {})
    st.session_state.custom_projects[proj_id] = record
    return proj_id


def update_project_assignments(project_id: str, assignments: dict) -> None:
    """
    Writes {role: employee_id} onto an already-saved custom project,
    after the mini-solver (item 11) picks a final staffing plan for
    it. Separate from save_project() because project creation is
    two-step: create the record first, run match_all_roles_adhoc() +
    the solver against it, then write the result back here.

    No-op (does nothing) if project_id isn't a known custom project --
    callers should have gotten this id from save_project() already.
    """
    _ensure_session_store()
    if project_id in st.session_state.custom_projects:
        st.session_state.custom_projects[project_id]["assignments"] = assignments


def save_candidate_pool(project_id: str, role_scores: dict) -> None:
    """
    Persists the full per-role candidate pool (role -> DataFrame,
    same shape match_all_roles_adhoc() returns) alongside a custom
    project's final assignment -- the ad-hoc equivalent of what
    score_matrix.csv already gives premade projects.

    Without this, staff_custom_project() computes the full ranked
    list (including the runner-up) during the solve, then throws it
    away -- only the winning assignment survives. This is what lets
    item 22's retrieve_adhoc() reconstruct runner-up comparisons and
    LLM explanations after the fact, instead of only at solve time.

    Called even on a partial/infeasible solve (some roles may still
    have scored candidates even if the overall solve failed) --
    storing whatever came back lets a later "why did this fail"
    explanation still have data to work with.
    """
    _ensure_session_store()
    st.session_state.custom_candidate_pools[project_id] = role_scores


def get_candidate_pool(project_id: str) -> dict | None:
    """
    Read side of save_candidate_pool(). Returns None if this project
    was never solved, or was solved in a session predating this item
    -- callers (dashboard.py) need to handle that gracefully rather
    than assume every custom project has one.
    """
    _ensure_session_store()
    return st.session_state.custom_candidate_pools.get(project_id)


def get_project_by_id(project_id: str,
                       projects_df: pd.DataFrame = None):
    """
    Looks up a project by id -- checks session-added custom projects
    first, then the real project list if projects_df is passed.
    Returns a dict, or None if not found in either place.
    """
    _ensure_session_store()
    if project_id in st.session_state.custom_projects:
        return st.session_state.custom_projects[project_id]
    if projects_df is not None and project_id in projects_df.index:
        return projects_df.loc[project_id].to_dict()
    return None


def load_all_projects(projects_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merges the real project list with session-added custom projects
    into one DataFrame, indexed by project_id -- same convention as
    projects_df. Mirrors employee_store.load_all_employees() exactly,
    so the dashboard's unified project selector (Phase 4 item 14) can
    read from a single source for both premade and custom projects.

    Note: "assignments" is a dict on custom project records and won't
    have a natural column equivalent in projects_df (that data lives
    in staffing_plan.csv separately for premade projects) -- it's
    dropped from the merged view here since callers that need
    assignments should go through get_busy_employee_ids() /
    get_project_by_id() instead, not read them off this merged table.
    """
    _ensure_session_store()
    custom = st.session_state.custom_projects
    if not custom:
        return projects_df

    # Drop "assignments" before building the DataFrame -- it's a
    # dict-per-row that doesn't fit a flat merged table and isn't
    # part of projects_df's schema anyway.
    custom_flat = {
        pid: {k: v for k, v in rec.items() if k != "assignments"}
        for pid, rec in custom.items()
    }
    custom_df = pd.DataFrame.from_dict(custom_flat, orient="index")
    custom_df.index.name = projects_df.index.name

    for col in projects_df.columns:
        if col not in custom_df.columns:
            custom_df[col] = None
    custom_df = custom_df[projects_df.columns]

    return pd.concat([projects_df, custom_df])


def list_custom_projects() -> list:
    """
    Session-added custom projects as a list of dicts. Used by
    get_busy_employee_ids() to walk every custom project's
    assignments alongside the premade staffing_plan.csv.
    """
    _ensure_session_store()
    return list(st.session_state.custom_projects.values())


def get_busy_employee_ids(staffing_plan_df: pd.DataFrame) -> set:
    """
    Phase 3 item 10.

    Every employee_id currently staffed anywhere -- premade projects
    (staffing_plan.csv, passed in by the caller, same "only one
    place reads the real path" convention as everywhere else) plus
    every custom project's assignments dict (session-scoped, so this
    always reflects the current session's state, not just what was
    true when the page loaded).

    This is the exclude_ids set that match_adhoc() /
    match_all_roles_adhoc() take -- call this once per matching pass
    so a new custom project's candidate pool never double-books
    someone already committed elsewhere this session.

    Note: staffing_plan.csv only has 14 rows covering 5 of the 30
    real projects (confirmed against the real file) -- most premade
    projects have no assignments at all yet, which is expected, not
    a bug. Employees on those un-staffed projects are NOT busy and
    should still show up as eligible candidates.

    Returns a set of employee_id strings -- both real (E0xx) and
    custom (CE0xx) ids can appear here, since match_adhoc() doesn't
    distinguish between them, it just excludes whatever's in the set.
    """
    busy = set(staffing_plan_df["employee_id"])

    for project in list_custom_projects():
        assignments = project.get("assignments", {})
        busy.update(assignments.values())

    return busy


def get_capacity_summary(project: dict,
                          employees_df: pd.DataFrame,
                          staffing_plan_df: pd.DataFrame,
                          min_avail: int = 60) -> dict:
    """
    Phase 3 item 12 -- the "X of 80 available" pre-check shown on the
    Create Project form, before the user commits to running the
    matcher/solver. Two things it deliberately does NOT do: it
    doesn't call the sentence-transformer model (this needs to be
    fast/cheap, unlike match_all_roles_adhoc()), and it doesn't check
    min_experience per role -- it's a capacity signal ("is the pool
    even large enough"), not a prediction of what the solver will
    return. A role can show 3 available here and still fail to solve
    if none of the 3 meet that role's specific min_experience --
    that's expected, this is a cheaper, coarser check upstream of it.

    project: dict with "required_roles" (";"-separated str) -- same
             shape match_all_roles_adhoc() expects.
    employees_df: the real roster, passed in by the caller (same
                  convention as everywhere else in this file).

    Returns:
        {
          "total_pool":      int,  # all employees, real + custom
          "total_available": int,  # not busy AND meets min_avail
          "by_role": {
              role: {"in_role": int, "available": int}, ...
          }
        }
    """
    all_employees = load_all_employees(employees_df)
    busy = get_busy_employee_ids(staffing_plan_df)

    free_mask = (
        (~all_employees.index.to_series().isin(busy)) &
        (all_employees["availability_pct"] >= min_avail)
    )

    summary = {
        "total_pool":      len(all_employees),
        "total_available": int(free_mask.sum()),
        "by_role":         {},
    }

    roles = [r.strip() for r in project["required_roles"].split(";")]
    for role in roles:
        in_role_mask = all_employees["role"] == role
        summary["by_role"][role] = {
            "in_role":   int(in_role_mask.sum()),
            "available": int((in_role_mask & free_mask).sum()),
        }

    return summary