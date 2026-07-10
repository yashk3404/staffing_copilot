"""
Employee storage interface -- Phase 2 item 6 (CE0xx ID scheme) and
the storage layer Phase 3 builds on.

Everything here talks to st.session_state right now. That's a
deliberate placeholder, not the final answer -- when persistence is
decided (Phase 5, items 19-20), only the *internals* of these
functions change. Every call site elsewhere (dashboard.py, matcher.py
via load_all_employees()) stays exactly the same.
"""

import streamlit as st
import pandas as pd

CUSTOM_ID_PREFIX = "CE"
CUSTOM_ID_WIDTH = 3  # CE001, CE002, ... -- never collides with the
                     # real E0xx range from employees_with_index.csv


def _ensure_session_store():
    if "custom_employees" not in st.session_state:
        st.session_state.custom_employees = {}      # {employee_id: dict}
    if "custom_employee_counter" not in st.session_state:
        st.session_state.custom_employee_counter = 0


def _next_custom_employee_id() -> str:
    _ensure_session_store()
    st.session_state.custom_employee_counter += 1
    n = st.session_state.custom_employee_counter
    return f"{CUSTOM_ID_PREFIX}{n:0{CUSTOM_ID_WIDTH}d}"


def save_employee(record: dict) -> str:
    """
    Commits a reviewed employee record -- from either the CV path
    (parse_resume() output, post-review-edit) or the manual form,
    both should be normalized to the same shape by the time they
    reach here -- into storage.

    Assigns a CE0xx id if the record doesn't already have one; never
    overwrites an existing id, so calling this twice on an edited
    record updates it in place instead of creating a duplicate.

    record is expected to have at least: name, role, experience_years,
    availability_pct, skills (list), department, location. Extra keys
    are stored as-is -- this function doesn't validate shape, that's
    the review form's job before it ever calls save_employee().

    Returns the assigned (or existing) employee_id.
    """
    _ensure_session_store()
    emp_id = record.get("employee_id")
    if not emp_id:
        emp_id = _next_custom_employee_id()
        record = {**record, "employee_id": emp_id}
    st.session_state.custom_employees[emp_id] = record
    return emp_id


def get_employee_by_id(employee_id: str,
                        employees_df: pd.DataFrame = None):
    """
    Looks up an employee by id -- checks session-added custom
    employees first, then the real roster if employees_df is passed.
    Returns a dict, or None if not found in either place.
    """
    _ensure_session_store()
    if employee_id in st.session_state.custom_employees:
        return st.session_state.custom_employees[employee_id]
    if employees_df is not None and employee_id in employees_df.index:
        return employees_df.loc[employee_id].to_dict()
    return None


def load_all_employees(employees_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merges the real roster with session-added custom employees into
    one DataFrame, indexed by employee_id (same index convention as
    employees_df -- matcher.py's set_index("employee_id")).

    This should become the single call site every candidate-pool /
    matching function reads from going forward -- match(), match_adhoc(),
    build_score_matrix(), etc. -- instead of reading employees_df
    directly. That's what makes "custom employees show up in premade
    projects' candidate pools too" (a decision made back in the
    original planning conversation) actually true everywhere at once,
    rather than needing to be threaded through each function one by
    one.
    """
    _ensure_session_store()
    custom = st.session_state.custom_employees
    if not custom:
        return employees_df

    custom_df = pd.DataFrame.from_dict(custom, orient="index")
    custom_df.index.name = employees_df.index.name

    # Custom employees may be missing columns the real roster has
    # (e.g. cost_band if that's not asked on the review form yet) --
    # fill with None rather than letting concat silently drop data
    # or raise on column mismatch.
    for col in employees_df.columns:
        if col not in custom_df.columns:
            custom_df[col] = None
    custom_df = custom_df[employees_df.columns]

    return pd.concat([employees_df, custom_df])


def list_custom_employees() -> list:
    """
    Session-added custom employees as a list of dicts -- this is
    exactly the shape find_possible_duplicates()'s custom_employees
    argument already expects, so Phase 3/4 wiring is just:
    find_possible_duplicates(candidate, employees_df, list_custom_employees()).
    """
    _ensure_session_store()
    return list(st.session_state.custom_employees.values())