"""
Employee storage interface -- Phase 2 item 6 (CE0xx ID scheme).

Item 23: internals swapped from st.session_state to Supabase Postgres
(employees table, item 21/23 schema). Every function signature and
return shape is unchanged from the session-state version -- callers
(dashboard.py, matcher.py via load_all_employees()) needed zero edits.

Scope (Option B, locked in the v3 roadmap): only user-added employees
live here. The 80-employee demo roster stays in
employees_with_index.csv + precomputed embeddings, untouched, and is
always passed in as employees_df by the caller.

Auth requirement: every function here needs an authenticated user in
st.session_state["user"] (set by auth.require_login()). Since
require_login() runs before any other UI in dashboard.py, that's
already guaranteed by the time these are called in the app -- tests
set up a fake user explicitly instead (see conftest.py).
"""

import streamlit as st
import pandas as pd

from src.auth import get_supabase_client

CUSTOM_ID_PREFIX = "CE"
CUSTOM_ID_WIDTH = 3  # CE001, CE002, ... -- never collides with the
                     # real E0xx range from employees_with_index.csv.
                     # Scoped per-owner as of item 23's 0002 migration,
                     # not globally unique -- two different users can
                     # each have their own "CE001".

_RECORD_COLUMNS = [
    "name", "role", "experience_years", "availability_pct",
    "skills", "department", "location", "github_username",
]


def _owner_id() -> str:
    user = st.session_state.get("user")
    if not user:
        raise RuntimeError(
            "employee_store called with no authenticated user in "
            "session -- auth.require_login() must run first."
        )
    return user.id


def _row_to_record(row: dict) -> dict:
    """DB row -> the dict shape callers already expect. skills comes
    back from Postgres as a jsonb list; matcher.py (and the rest of
    the app) expects the same ';'-joined string convention
    employees_with_index.csv uses, so it's converted back here --
    this is the one place that difference should live."""
    record = dict(row)
    skills = record.get("skills")
    if isinstance(skills, list):
        record["skills"] = ";".join(skills)
    return record


def _record_to_row(record: dict, owner_id: str, employee_id: str) -> dict:
    skills = record.get("skills", "")
    if isinstance(skills, str):
        skills_list = [s.strip() for s in skills.split(";") if s.strip()]
    else:
        skills_list = list(skills) if skills else []
    return {
        "employee_id": employee_id,
        "owner_id": owner_id,
        "name": record.get("name"),
        "role": record.get("role"),
        "experience_years": record.get("experience_years"),
        "availability_pct": record.get("availability_pct"),
        "skills": skills_list,
        "department": record.get("department"),
        "location": record.get("location"),
        "github_username": record.get("github_username"),
    }


def _next_custom_employee_id(supabase, owner_id: str) -> str:
    """Sequential per-owner, not globally unique -- see the
    CUSTOM_ID_PREFIX comment and the 0002 migration. Computed from
    this owner's own rows (the only ones RLS lets them see), so it's
    always a fresh max+1 rather than a counter that could drift from
    what's actually stored."""
    result = (
        supabase.table("employees")
        .select("employee_id")
        .eq("owner_id", owner_id)
        .execute()
    )
    nums = []
    for row in result.data:
        eid = row["employee_id"]
        if eid.startswith(CUSTOM_ID_PREFIX) and eid[len(CUSTOM_ID_PREFIX):].isdigit():
            nums.append(int(eid[len(CUSTOM_ID_PREFIX):]))
    n = (max(nums) + 1) if nums else 1
    return f"{CUSTOM_ID_PREFIX}{n:0{CUSTOM_ID_WIDTH}d}"


def save_employee(record: dict) -> str:
    """
    Commits a reviewed employee record -- from either the CV path
    (parse_resume() output, post-review-edit) or the manual form --
    into Supabase, scoped to the logged-in user via owner_id.

    Assigns a CE0xx id if the record doesn't already have one; calling
    this again with an existing employee_id upserts in place instead
    of creating a duplicate (matches the old session-dict behavior).

    Returns the assigned (or existing) employee_id.
    """
    supabase = get_supabase_client()
    owner_id = _owner_id()

    emp_id = record.get("employee_id")
    if not emp_id:
        emp_id = _next_custom_employee_id(supabase, owner_id)

    row = _record_to_row(record, owner_id, emp_id)
    supabase.table("employees").upsert(
        row, on_conflict="employee_id,owner_id"
    ).execute()
    return emp_id


def get_employee_by_id(employee_id: str,
                        employees_df: pd.DataFrame = None):
    """
    Looks up an employee by id -- checks this user's Supabase-stored
    custom employees first, then the real roster if employees_df is
    passed. Returns a dict, or None if not found in either place.
    """
    supabase = get_supabase_client()
    owner_id = _owner_id()

    result = (
        supabase.table("employees")
        .select("*")
        .eq("employee_id", employee_id)
        .eq("owner_id", owner_id)
        .limit(1)
        .execute()
    )
    if result.data:
        return _row_to_record(result.data[0])
    if employees_df is not None and employee_id in employees_df.index:
        return employees_df.loc[employee_id].to_dict()
    return None


def load_all_employees(employees_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merges the real roster with this user's Supabase-stored custom
    employees into one DataFrame, indexed by employee_id (same index
    convention as employees_df).

    This is the single call site every candidate-pool / matching
    function reads from -- match(), match_adhoc(),
    build_score_matrix(), etc. -- instead of reading employees_df
    directly, so custom employees show up as candidates everywhere.
    """
    supabase = get_supabase_client()
    owner_id = _owner_id()

    result = supabase.table("employees").select("*").eq("owner_id", owner_id).execute()
    if not result.data:
        return employees_df

    custom_records = [_row_to_record(r) for r in result.data]
    custom_df = pd.DataFrame(custom_records).set_index("employee_id")
    custom_df.index.name = employees_df.index.name

    # Custom employees may be missing columns the real roster has
    # (e.g. cost_band if that's not asked on the review form) -- fill
    # with None rather than letting concat silently drop data or
    # raise on column mismatch. Also drops owner_id/created_at/id,
    # which aren't part of employees_df's schema -- matches the old
    # session-state version's merged view exactly.
    for col in employees_df.columns:
        if col not in custom_df.columns:
            custom_df[col] = None
    custom_df = custom_df[employees_df.columns]

    return pd.concat([employees_df, custom_df])


def list_custom_employees() -> list:
    """
    This user's Supabase-stored custom employees as a list of dicts --
    the shape find_possible_duplicates()'s custom_employees argument
    already expects.
    """
    supabase = get_supabase_client()
    owner_id = _owner_id()
    result = supabase.table("employees").select("*").eq("owner_id", owner_id).execute()
    return [_row_to_record(r) for r in result.data]