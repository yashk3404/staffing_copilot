"""
Project storage interface -- Phase 3 item 9 (C0xx ID scheme).

Item 23: internals swapped from st.session_state to Supabase Postgres
(projects + assignments tables, item 21/23 schema). Every function
signature and return shape is unchanged from the session-state
version -- callers (dashboard.py, matcher.match_adhoc(),
get_busy_employee_ids()) needed zero edits.

Item 25-adjacent fix (migration 0003): save_candidate_pool()/
get_candidate_pool() moved OFF st.session_state and into Postgres too
-- assignments.final_score + projects.candidate_pool (jsonb). They
used to be session-scoped on purpose (Option B's schema didn't cover
them yet), which meant match scores, runner-up comparisons, and the
Full Candidate Pool table all silently went blank on a refresh or a
new session. Requires migration 0003 to be applied -- see that file's
comment for exactly what it adds.

Schema for the parts that ARE in Postgres:
    projects:    project_id, project_name, client, required_roles
                 (";"-sep), required_skills (";"-sep), min_experience,
                 deadline_days, budget_band, priority,
                 candidate_pool (jsonb, migration 0003)
    assignments: project_id, role, employee_id, final_score (migration
                 0003) -- one row per (project_id, role); reconstructed
                 into the {role: employee_id} dict shape callers
                 expect (scores are fetched separately via
                 get_assignment_scores() to keep that shape unchanged).
"""

import pandas as pd

from src.employee_store import _owner_id
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


def update_project_assignments(project_id: str, assignments: dict,
                                scores: dict = None) -> None:
    """
    Writes {role: employee_id} as rows in the assignments table, after
    the mini-solver picks a final staffing plan for a custom project.
    Replaces (delete then insert) rather than upserts per-role, so a
    re-solve that drops a role doesn't leave a stale assignment
    behind.

    scores: optional {role: final_score} (migration 0003) -- written
    alongside each row so Role Details' "score: X" and the Assigned
    Team cards can read a real number straight off the assignment
    itself, without needing get_candidate_pool() (which may be None
    for a project solved before migration 0003, or never re-solved
    since). None (the default) leaves final_score null, same as
    before this param existed.

    No-op if project_id isn't a known project owned by this user --
    same silent-no-op contract the session-state version had. This is
    checked explicitly up front rather than left to the DB's foreign
    key constraint, so an unknown id fails quietly instead of
    surfacing a raw FK error to the caller.
    """
    supabase = get_supabase_client()
    owner_id = _owner_id()
    scores = scores or {}

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
                "final_score": scores.get(role),
            }
            for role, emp_id in assignments.items()
        ]
        supabase.table("assignments").insert(rows).execute()


def get_assignment_scores(project_id: str) -> dict:
    """
    {role: final_score} for one of this user's custom projects
    (migration 0003). Kept separate from get_project_by_id()'s
    "assignments" dict (which stays {role: employee_id}, unchanged
    shape) rather than folding scores in there, since several call
    sites already destructure assignments.items() as (role, emp_id)
    pairs -- changing that shape would've meant touching every one of
    them for a purely additive feature. Missing/null scores (an
    assignment written before migration 0003, or via a scores=None
    call) come back as None for that role, not KeyError.
    """
    supabase = get_supabase_client()
    owner_id = _owner_id()
    result = (
        supabase.table("assignments")
        .select("role, final_score")
        .eq("project_id", project_id)
        .eq("owner_id", owner_id)
        .execute()
    )
    return {row["role"]: row.get("final_score") for row in result.data}


def delete_project(project_id: str) -> None:
    """
    Fully removes one of this user's own custom projects -- the
    project row AND its assignments (children delete via `on delete
    cascade` in the assignments FK, but assignments is deleted
    explicitly first anyway so this doesn't depend on that cascade
    being configured correctly).

    This is what "End Project" in the dashboard calls: unlike
    update_project_assignments(project_id, {}) (which only un-staffs
    and leaves the empty project sitting in the picker), this removes
    the project from load_all_projects() / list_custom_projects()
    entirely -- so it disappears from the "Select Project" dropdown,
    matching what "ended" should mean.

    Scoped by owner_id in the query AND enforced again by the
    projects_delete / assignments_delete RLS policies at the DB level.
    Silent no-op if project_id doesn't belong to this user.
    """
    supabase = get_supabase_client()
    owner_id = _owner_id()

    supabase.table("assignments").delete().eq("project_id", project_id).eq(
        "owner_id", owner_id
    ).execute()
    supabase.table("projects").delete().eq("project_id", project_id).eq(
        "owner_id", owner_id
    ).execute()


def save_candidate_pool(project_id: str, role_scores: dict) -> None:
    """
    Persists the full per-role candidate pool (role -> DataFrame) to
    Postgres (projects.candidate_pool, migration 0003) so runner-up
    comparisons and LLM explanations for a custom project survive a
    refresh or a brand new session, not just the request that solved
    it.

    Stores a trimmed column set per candidate rather than every
    intermediate scoring feature (semantic_score, availability_factor,
    etc.), to keep the JSON payload small -- exactly the columns
    retrieve_context.retrieve_adhoc() and dashboard.py's Full
    Candidate Pool table actually read: employee_id, name, role,
    experience_years, availability_pct, skills, final_score, eligible.

    No-op (raises nothing, writes nothing) if project_id isn't a known
    project owned by this user -- the update's .eq() filters simply
    match zero rows, same silent-no-op contract every other write in
    this module already has.
    """
    supabase = get_supabase_client()
    owner_id = _owner_id()

    keep_cols = ["employee_id", "name", "role", "experience_years",
                 "availability_pct", "skills", "final_score", "eligible"]
    payload = {
        role: df[[c for c in keep_cols if c in df.columns]].to_dict("records")
        for role, df in role_scores.items()
    }
    supabase.table("projects").update(
        {"candidate_pool": payload}
    ).eq("project_id", project_id).eq("owner_id", owner_id).execute()


def get_candidate_pool(project_id: str):
    """
    Read side of save_candidate_pool(). Returns None if this project
    was never solved, was created/solved before migration 0003 landed,
    or doesn't belong to this user -- callers (dashboard.py,
    retrieve_context.py) already handle a None pool gracefully.
    """
    supabase = get_supabase_client()
    owner_id = _owner_id()

    result = (
        supabase.table("projects")
        .select("candidate_pool")
        .eq("project_id", project_id)
        .eq("owner_id", owner_id)
        .limit(1)
        .execute()
    )
    if not result.data or not result.data[0].get("candidate_pool"):
        return None

    raw = result.data[0]["candidate_pool"]
    return {role: pd.DataFrame(records) for role, records in raw.items()}


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
                          candidate_pool_df: pd.DataFrame,
                          staffing_plan_df: pd.DataFrame,
                          min_avail: int = 60) -> dict:
    """
    Phase 3 item 12 -- the "X of 80 available" pre-check shown on the
    Create Project form.

    candidate_pool_df: the ALREADY-FINAL pool to check against --
    dashboard.py passes employee_store.load_own_employees(employees)
    for custom projects (v3: no synthetic/demo candidates), so this
    function no longer calls load_all_employees() internally. Doing
    that merge here used to be convenient when there was only one
    possible pool; now that custom projects and Demo Mode use two
    different pools, merging inside this function would silently
    double-count rows if a caller ever passed something already
    merged. Callers own the merge decision; this just does the count.

    Returns:
        {
          "total_pool":      int,  # size of candidate_pool_df
          "total_available": int,  # not busy AND meets min_avail
          "by_role": {
              role: {"in_role": int, "available": int}, ...
          }
        }
    """
    busy = get_busy_employee_ids(staffing_plan_df)

    free_mask = (
        (~candidate_pool_df.index.to_series().isin(busy)) &
        (candidate_pool_df["availability_pct"] >= min_avail)
    )

    summary = {
        "total_pool":      len(candidate_pool_df),
        "total_available": int(free_mask.sum()),
        "by_role":         {},
    }

    roles = [r.strip() for r in project["required_roles"].split(";")]
    for role in roles:
        in_role_mask = candidate_pool_df["role"] == role
        summary["by_role"][role] = {
            "in_role":   int(in_role_mask.sum()),
            "available": int((in_role_mask & free_mask).sum()),
        }

    return summary