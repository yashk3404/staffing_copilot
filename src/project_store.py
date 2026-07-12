"""
Project storage interface -- Phase 3 item 9 (C0xx ID scheme).

Item 23: internals swapped from st.session_state to Supabase Postgres
(projects + assignments tables, item 21/23 schema). Every function
signature and return shape is unchanged from the session-state
version -- callers (dashboard.py, matcher.match_adhoc(),
get_busy_employee_ids()) needed zero edits.

One deliberate scope note: save_candidate_pool()/get_candidate_pool()
stay session-scoped, NOT moved to Supabase. The v3 roadmap's Option B
only scoped employees/projects (and, as of item 23, their
assignments) into Postgres -- there's no schema for the full ranked
candidate pool (role -> DataFrame), and adding one wasn't part of
item 21/23's design. Practical effect: runner-up comparisons /
explanation context for a custom project still won't survive a
browser refresh, same limitation item 22 already accepted for login
itself. Worth a deliberate v4-ish follow-up if that turns out to
matter, not a silent gap.

Schema for the parts that ARE in Postgres:
    projects:    project_id, project_name, client, required_roles
                 (";"-sep), required_skills (";"-sep), min_experience,
                 deadline_days, budget_band, priority
    assignments: project_id, role, employee_id -- one row per
                 (project_id, role); reconstructed into the
                 {role: employee_id} dict shape callers expect.
"""

import streamlit as st
import pandas as pd

from src.employee_store import load_all_employees, _owner_id
from src.auth import get_supabase_client

CUSTOM_ID_PREFIX = "C"
CUSTOM_ID_WIDTH = 3  # C001, C002, ... -- never collides with the real
                     # P0xx range from projects_with_index.csv.
                     # Scoped per-owner (item 23's 0002 migration),
                     # not globally unique.


def _next_custom_project_id(supabase, owner_id: str) -> str:
    result = (
        supabase.table("projects")
        .select("project_id")
        .eq("owner_id", owner_id)
        .execute()
    )
    nums = []
    for row in result.data:
        pid = row["project_id"]
        if pid.startswith(CUSTOM_ID_PREFIX) and pid[len(CUSTOM_ID_PREFIX):].isdigit():
            nums.append(int(pid[len(CUSTOM_ID_PREFIX):]))
    n = (max(nums) + 1) if nums else 1
    return f"{CUSTOM_ID_PREFIX}{n:0{CUSTOM_ID_WIDTH}d}"


def _fetch_assignments_for_project(supabase, owner_id: str, project_id: str) -> dict:
    result = (
        supabase.table("assignments")
        .select("role, employee_id")
        .eq("project_id", project_id)
        .eq("owner_id", owner_id)
        .execute()
    )
    return {row["role"]: row["employee_id"] for row in result.data}


def save_project(record: dict) -> str:
    """
    Commits a project record -- from the "Create Project" form,
    post-review-edit -- into Supabase, scoped to the logged-in user.

    Assigns a C0xx id if the record doesn't already have one; calling
    this again with an existing project_id upserts in place instead of
    creating a duplicate.

    "assignments" is no longer stored on this record -- it's derived
    from the assignments table on read (get_project_by_id(),
    list_custom_projects()), always starting empty for a brand new
    project until update_project_assignments() runs.

    Returns the assigned (or existing) project_id.
    """
    supabase = get_supabase_client()
    owner_id = _owner_id()

    proj_id = record.get("project_id")
    if not proj_id:
        proj_id = _next_custom_project_id(supabase, owner_id)

    row = {
        "project_id": proj_id,
        "owner_id": owner_id,
        "project_name": record.get("project_name"),
        "client": record.get("client"),
        "required_roles": record.get("required_roles"),
        "required_skills": record.get("required_skills"),
        "min_experience": record.get("min_experience"),
        "deadline_days": record.get("deadline_days"),
        "budget_band": record.get("budget_band"),
        "priority": record.get("priority"),
    }
    supabase.table("projects").upsert(
        row, on_conflict="project_id,owner_id"
    ).execute()
    return proj_id


def update_project_assignments(project_id: str, assignments: dict) -> None:
    """
    Writes {role: employee_id} as rows in the assignments table, after
    the mini-solver picks a final staffing plan for a custom project.
    Replaces (delete then insert) rather than upserts per-role, so a
    re-solve that drops a role doesn't leave a stale assignment
    behind.

    No-op if project_id isn't a known project owned by this user --
    same silent-no-op contract the session-state version had. This is
    checked explicitly up front rather than left to the DB's foreign
    key constraint, so an unknown id fails quietly instead of
    surfacing a raw FK error to the caller.
    """
    supabase = get_supabase_client()
    owner_id = _owner_id()

    exists = (
        supabase.table("projects")
        .select("project_id")
        .eq("project_id", project_id)
        .eq("owner_id", owner_id)
        .limit(1)
        .execute()
    )
    if not exists.data:
        return

    supabase.table("assignments").delete().eq("project_id", project_id).eq(
        "owner_id", owner_id
    ).execute()

    if assignments:
        rows = [
            {
                "project_id": project_id,
                "role": role,
                "employee_id": emp_id,
                "owner_id": owner_id,
            }
            for role, emp_id in assignments.items()
        ]
        supabase.table("assignments").insert(rows).execute()


def save_candidate_pool(project_id: str, role_scores: dict) -> None:
    """
    Session-scoped, deliberately not moved to Supabase -- see the
    module docstring. Persists the full per-role candidate pool
    (role -> DataFrame) alongside a custom project's final assignment,
    so retrieve_adhoc() can reconstruct runner-up comparisons and LLM
    explanations after the fact within the same session.
    """
    if "custom_candidate_pools" not in st.session_state:
        st.session_state.custom_candidate_pools = {}
    st.session_state.custom_candidate_pools[project_id] = role_scores


def get_candidate_pool(project_id: str) -> dict | None:
    """Read side of save_candidate_pool(). Returns None if this
    project was never solved this session -- callers (dashboard.py)
    need to handle that gracefully."""
    if "custom_candidate_pools" not in st.session_state:
        st.session_state.custom_candidate_pools = {}
    return st.session_state.custom_candidate_pools.get(project_id)


def get_project_by_id(project_id: str,
                       projects_df: pd.DataFrame = None):
    """
    Looks up a project by id -- checks this user's Supabase-stored
    custom projects first (with assignments reconstructed from the
    assignments table), then the real project list if projects_df is
    passed. Returns a dict, or None if not found in either place.
    """
    supabase = get_supabase_client()
    owner_id = _owner_id()

    result = (
        supabase.table("projects")
        .select("*")
        .eq("project_id", project_id)
        .eq("owner_id", owner_id)
        .limit(1)
        .execute()
    )
    if result.data:
        record = dict(result.data[0])
        record["assignments"] = _fetch_assignments_for_project(
            supabase, owner_id, project_id
        )
        return record
    if projects_df is not None and project_id in projects_df.index:
        return projects_df.loc[project_id].to_dict()
    return None


def load_all_projects(projects_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merges the real project list with this user's Supabase-stored
    custom projects into one DataFrame, indexed by project_id.
    "assignments" is never a column here (it wasn't in the old
    session-state version's merged view either) -- callers that need
    assignments should go through get_busy_employee_ids() /
    get_project_by_id() instead.
    """
    supabase = get_supabase_client()
    owner_id = _owner_id()

    result = supabase.table("projects").select("*").eq("owner_id", owner_id).execute()
    if not result.data:
        return projects_df

    custom_df = pd.DataFrame(result.data).set_index("project_id")
    custom_df.index.name = projects_df.index.name

    for col in projects_df.columns:
        if col not in custom_df.columns:
            custom_df[col] = None
    custom_df = custom_df[projects_df.columns]

    return pd.concat([projects_df, custom_df])


def list_custom_projects() -> list:
    """
    This user's Supabase-stored custom projects as a list of dicts,
    each with "assignments" reconstructed from the assignments table.
    Used by get_busy_employee_ids() to walk every custom project's
    assignments alongside the premade staffing_plan.csv.

    Fetches all of this user's projects and all of this user's
    assignments in two queries total (not one query per project) --
    matters once someone has more than a couple custom projects.
    """
    supabase = get_supabase_client()
    owner_id = _owner_id()

    projects_result = supabase.table("projects").select("*").eq("owner_id", owner_id).execute()
    assignments_result = (
        supabase.table("assignments").select("*").eq("owner_id", owner_id).execute()
    )

    by_project = {}
    for row in assignments_result.data:
        by_project.setdefault(row["project_id"], {})[row["role"]] = row["employee_id"]

    projects = []
    for row in projects_result.data:
        record = dict(row)
        record["assignments"] = by_project.get(record["project_id"], {})
        projects.append(record)
    return projects


def get_busy_employee_ids(staffing_plan_df: pd.DataFrame) -> set:
    """
    Phase 3 item 10. Every employee_id currently staffed anywhere --
    premade projects (staffing_plan.csv, passed in by the caller) plus
    every one of this user's custom projects' assignments (now read
    from Supabase via list_custom_projects(), always reflecting the
    current DB state rather than just what was true this session).

    This is the exclude_ids set that match_adhoc() /
    match_all_roles_adhoc() take.

    Returns a set of employee_id strings -- both real (E0xx) and
    custom (CE0xx) ids can appear here.
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
    Create Project form. Unchanged from the session-state version --
    it only calls load_all_employees() / get_busy_employee_ids(),
    both of which now read Supabase underneath without this function
    needing to know that.

    project: dict with "required_roles" (";"-separated str).
    employees_df: the real roster, passed in by the caller.

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