"""
Staffing Copilot — Streamlit Dashboard v2
Adds LLM explanation panels and "Why not X?" runner-up view.

Run:
    venv\\Scripts\\python.exe -m streamlit run src/dashboard.py
"""

import os
import sys
import tempfile
import streamlit as st
import pandas as pd
from pathlib import Path

# Bridge Streamlit Cloud's secrets into an env var so
# generate_explanation.py's Groq fallback can find it, both locally
# (via .env) and when deployed (via st.secrets). Safe no-op if the
# key isn't set anywhere.
try:
    if "GROQ_API_KEY" in st.secrets:
        os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]
except Exception:
    pass  
sys.path.append(str(Path(__file__).parent))
# Also add the repo root (not just src/) so absolute "src.X" imports
# used by project_store.py / optimize_staffing.py / employee_store.py
# resolve regardless of how this script is launched -- run_dashboard.bat's
# `python -m streamlit run src/dashboard.py` happens to put the repo
# root on sys.path via -m semantics already, but that's an invocation
# detail, not something this file should silently depend on.
sys.path.append(str(Path(__file__).parent.parent))
from retrieve_context import ContextRetriever
from generate_explanation import generate_explanation
from matcher import Matcher
from src.project_store import (
    save_project,
    get_project_by_id,
    get_capacity_summary,
    load_all_projects,
    get_candidate_pool,
    get_busy_employee_ids,
    list_custom_projects,
    delete_project,
)
from src.optimize_staffing import staff_custom_project, StaffingOptimizer
from src.employee_store import (
    save_employee,
    list_custom_employees,
    load_all_employees,
    load_own_employees,
    delete_employee,
)
from src.resume_parser import (
    parse_resume,
    suggest_skill_matches,
    suggest_role,
    find_possible_duplicates,
    format_duplicate_warning,
)

BASE = Path(__file__).parent.parent / "data" / "processed"

st.set_page_config(
    page_title="Staffing Copilot",
    page_icon="🧠",
    layout="wide",
)

# Item 22 -- everything below this line assumes an authenticated
# user. require_login() st.stop()s the script (showing login/signup
# instead) until st.session_state["user"] is set.
from src.auth import require_login
require_login()

# ── Item 18 -- visual polish ─────────────────────────────────────────
#
# One shared <style> block, injected once here, plus small HTML
# builders (badge / chip) reused by every section below that used to
# be a flat st.metric() or plain-text label -- Project Details and
# Assigned Team both read off the same palette now instead of each
# section inventing its own. Colors use translucent/rgba backgrounds
# rather than solid white/black so cards stay legible on both light
# and dark Streamlit themes.

st.markdown("""
<style>
.sc-card {
    background: rgba(127, 127, 127, 0.06);
    border: 1px solid rgba(127, 127, 127, 0.18);
    border-left: 4px solid #6C63FF;
    border-radius: 10px;
    padding: 14px 18px;
    margin-bottom: 12px;
}
.sc-card-title {
    font-weight: 600;
    font-size: 1.05rem;
    margin-bottom: 2px;
}
.sc-card-sub {
    opacity: 0.65;
    font-size: 0.85rem;
    margin-bottom: 10px;
}
.sc-row {
    display: flex;
    flex-wrap: wrap;
    gap: 18px;
    font-size: 0.88rem;
    opacity: 0.9;
}
.sc-badge {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 999px;
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.02em;
    color: #1a1a1a;
    margin-right: 6px;
}
.sc-chip {
    display: inline-block;
    padding: 2px 9px;
    border-radius: 6px;
    font-size: 0.78rem;
    background: rgba(127, 127, 127, 0.14);
    margin: 2px 4px 2px 0;
}
</style>
""", unsafe_allow_html=True)

# Priority runs low -> critical; budget runs low -> high. Both follow
# the plan's "red/yellow/green" anchor (green = low/cheap, red = the
# thing that needs attention), with an amber/orange step in between
# so all four priority levels -- not just the three named in the plan
# -- get a distinct color rather than two of them sharing one.
PRIORITY_COLORS = {
    "low":      "#8BD17C",
    "medium":   "#F5D061",
    "high":     "#F2A65A",
    "critical": "#F16A6A",
}
BUDGET_COLORS = {
    "low":    "#8BD17C",
    "medium": "#F5D061",
    "high":   "#F16A6A",
}


def _badge(label: str, color: str) -> str:
    return f'<span class="sc-badge" style="background:{color}">{label}</span>'


def priority_badge(value) -> str:
    v = str(value).lower()
    return _badge(str(value).title() or "—", PRIORITY_COLORS.get(v, "#B0B0B0"))


def budget_badge(value) -> str:
    v = str(value).lower()
    return _badge(str(value).title() or "—", BUDGET_COLORS.get(v, "#B0B0B0"))

# ── Load data ─────────────────────────────────────────────────────

@st.cache_resource
def load_retriever():
    return ContextRetriever(str(BASE))

@st.cache_resource
def load_matcher():
    # Cached per session (@st.cache_resource, not @st.cache_data) since
    # this holds a live SentenceTransformer model + embeddings, same
    # reasoning as load_retriever() -- expensive to reload per rerun.
    m = Matcher()
    m.load()
    return m

@st.cache_data
def load_plan():
    return pd.read_csv(BASE / "staffing_plan.csv")

@st.cache_data
def load_score_matrix():
    return pd.read_csv(BASE / "score_matrix.csv")

@st.cache_data
def load_employees():
    return pd.read_csv(
        BASE / "employees_with_index.csv"
    ).set_index("employee_id")

@st.cache_data
def load_projects():
    return pd.read_csv(
        BASE / "projects.csv"
    ).set_index("project_id")

retriever    = load_retriever()
plan         = load_plan()
score_matrix = load_score_matrix()
employees    = load_employees()
projects_df  = load_projects()

# ── Item 17 -- merge custom employees into existing views ───────────
#
# load_all_employees() concats the real roster with any session-added
# CE0xx employees (employee_store.py). This is the single merge point
# every *display* call site below reads through instead of `employees`
# directly, so a custom employee now shows up in the Assigned Team
# table, Role Details panels, and the Create/Add Employee role
# dropdowns -- not just in get_employee_by_id() / list_custom_employees()
# like before.
#
# Kept as a SEPARATE variable rather than reassigning `employees`, for
# two reasons:
#   1. get_capacity_summary() already calls load_all_employees()
#      internally on whatever employees_df it's given -- passing it
#      the already-merged frame would concat the custom employees in
#      twice.
#   2. find_possible_duplicates() below takes the real roster and
#      list_custom_employees() as two separate arguments by design --
#      passing it a pre-merged frame would check every custom
#      employee against itself and double up warnings.
# Not @st.cache_data'd: it reads live st.session_state, which changes
# every time an employee gets added.
all_employees_df = load_all_employees(employees)

# v3 UI update -- custom (C0xx) projects are "your" projects, so they
# should only ever be staffed from "your" employee list, never the
# shared 80-person demo/synthetic roster. own_employees_df is that
# restricted pool -- see load_own_employees()'s docstring. Kept
# separate from all_employees_df (which stays the merged view used
# everywhere a *display* just needs to resolve an employee_id to a
# name, e.g. an already-assigned custom-project team member who might
# technically be a demo-roster id from before this change).
own_employees_df = load_own_employees(employees)

# ── Popup dialogs (End Project / Delete Employee) ────────────────────
#
# st.dialog opens a real modal popup -- used here for (a) confirming a
# destructive action before it happens, and (b) a one-shot "done"
# notice right after it happens. The "done" notice is driven by a
# session_state flag set immediately before the st.rerun() that
# follows a delete, and popped here (so it only shows once) at the top
# of the very next run, before any mode-specific UI renders.

@st.dialog("End Project")
def _confirm_end_project(project_id: str, project_name: str, n_assigned: int):
    st.write(
        f"This will un-staff **{n_assigned}** employee(s) and "
        f"permanently remove **{project_id} — {project_name}** from "
        f"your project list. This can't be undone."
    )
    c1, c2 = st.columns(2)
    if c1.button("🛑 End Project", type="primary", width="stretch"):
        delete_project(project_id)
        st.session_state["_project_ended_notice"] = project_name
        st.session_state["_pending_mode"] = "Browse Projects"
        st.session_state.pop("_pending_project_select", None)
        st.rerun()
    if c2.button("Cancel", width="stretch"):
        st.rerun()


@st.dialog("Delete Employee")
def _confirm_delete_employee(employee_id: str, name: str, is_busy: bool):
    st.write(
        f"Permanently delete **{name}** ({employee_id}) from your "
        f"employee list? This can't be undone."
    )
    if is_busy:
        st.warning(
            "⚠️ This employee is currently assigned to a project. "
            "Deleting them leaves that assignment pointing at a "
            "removed employee -- consider ending that project first."
        )
    c1, c2 = st.columns(2)
    if c1.button("🗑️ Delete", type="primary", width="stretch"):
        delete_employee(employee_id)
        st.session_state["_employee_deleted_notice"] = name
        st.rerun()
    if c2.button("Cancel", width="stretch"):
        st.rerun()


if "_project_ended_notice" in st.session_state:
    _ended_name = st.session_state.pop("_project_ended_notice")

    @st.dialog("Project Ended")
    def _project_ended_popup(name: str):
        st.success(f"✅ **{name}** was ended and removed from your project list.")
        if st.button("OK", type="primary", width="stretch"):
            st.rerun()

    _project_ended_popup(_ended_name)

if "_employee_deleted_notice" in st.session_state:
    st.toast(
        f"🗑️ Deleted employee: {st.session_state.pop('_employee_deleted_notice')}"
    )

# ── Sidebar: mode switch (item 13) ──────────────────────────────────
#
# This is the structural change the rest of Phase 4 hangs off of.
# Everything that used to live on one long-scrolling page now renders
# under exactly one of three modes. "Browse Projects" carries the old
# main-page content (project detail view). "Create Project" carries
# what used to be the always-open "➕ Create New Project" expander,
# now a full mode instead of a collapsible section. "Add Employee" is
# a stub for item 15 -- not built yet, flagged as such below rather
# than silently missing.

st.sidebar.title("🧠 Staffing Copilot")
st.sidebar.markdown("---")

# Promote a pending mode switch (see the "Add New" sentinel handling
# and the post-creation rerun below) into the radio's own bound key.
# Same constraint as project_selector below: this MUST happen before
# the radio is instantiated, or Streamlit raises StreamlitAPIException
# for mutating a keyed widget's state after it's already rendered once
# in a given run.
if "_pending_mode" in st.session_state:
    st.session_state["app_mode"] = st.session_state.pop("_pending_mode")

mode = st.sidebar.radio(
    "Mode",
    ["🎬 Demo Mode", "Browse Projects", "Create Project",
     "👥 My Employees", "Add Employee"],
    key="app_mode",
)

st.sidebar.markdown("---")

st.sidebar.markdown("---")

# ── Item 14 -- project selector, reading through load_all_projects()
# (item 9) instead of only projects_df.
#
# Rescoped: Browse Projects now lists ONLY custom (C0xx) projects --
# real/uploaded data the user actually created via Create Project.
# Premade (P0xx) projects moved entirely to 🎬 Demo Mode, which has
# its own selector + live-solve + view flow; showing them here too
# would just be a second, staler way to reach the same 30 projects
# (this dropdown never lets you *run* a premade project anyway, only
# view an existing staffing_plan.csv assignment).
all_projects_df = load_all_projects(projects_df)

custom_ids  = set(all_projects_df.index) - set(projects_df.index)
project_ids = sorted(custom_ids)

NEW_PROJECT_SENTINEL = "__new_project__"

selected_project = None
is_custom_project = False

if mode == "Browse Projects":
    # Promote a pending post-creation selection (see the st.rerun()
    # call in Create Project mode below) into the selectbox's own
    # bound key -- must happen here, before the selectbox is
    # instantiated, for the same reason as the mode promotion above.
    if "_pending_project_select" in st.session_state:
        st.session_state["project_selector"] = \
            st.session_state.pop("_pending_project_select")

    dropdown_options = project_ids + [NEW_PROJECT_SENTINEL]

    def _project_label(pid):
        if pid == NEW_PROJECT_SENTINEL:
            return "➕ Add New Project"
        name = all_projects_df.loc[pid, "project_name"] \
            if pid in all_projects_df.index else ""
        return f"{pid} — {name}" if name else pid

    if not project_ids:
        st.sidebar.caption(
            "No projects created yet -- use **➕ Add New Project** "
            "below, or head to **🎬 Demo Mode** to try the built-in "
            "premade projects."
        )

    selected_project = st.sidebar.selectbox(
        "Select Project",
        dropdown_options,
        format_func=_project_label,
        key="project_selector",
    )

    if selected_project == NEW_PROJECT_SENTINEL:
        # The sentinel is a UI trigger, not a real selection -- jump
        # to Create Project mode, and reset the dropdown back to a
        # real project so it isn't left stuck on the sentinel the
        # next time Browse Projects mode renders. Can't set
        # "project_selector" or "app_mode" directly here (both
        # already instantiated earlier in this run) -- stash under
        # the "_pending_*" keys instead, promoted at the top of the
        # next run before those widgets exist yet.
        fallback = project_ids[0] if project_ids else NEW_PROJECT_SENTINEL
        st.session_state["_pending_project_select"] = fallback
        st.session_state["_pending_mode"] = "Create Project"
        st.rerun()

    # Always true in practice now (project_ids is custom-only), kept
    # as a variable rather than hardcoded True since the render
    # section below still branches on it -- one flag, one meaning,
    # even though the premade branch is now unreachable from here.
    is_custom_project = selected_project in custom_ids

st.sidebar.markdown("---")

_custom_total_assignments = sum(
    len(p.get("assignments", {})) for p in list_custom_projects()
)
st.sidebar.markdown(
    f"**My employees:** {len(own_employees_df)}  \n"
    f"**Custom projects:** {len(custom_ids)}  \n"
    f"**Custom assignments:** {_custom_total_assignments}"
)

# ── 🎬 Demo Mode support ─────────────────────────────────────────────
#
# Demo Mode live-solves premade projects into st.session_state, never
# into staffing_plan.csv, so retriever.retrieve() (which reads
# self.plan) can't see these assignments. Reuses retrieve_adhoc()
# instead -- it was built for custom (C0xx) projects, but all it
# actually needs is a project-shaped dict with "assignments" and a
# {role: DataFrame} candidate pool, neither of which is
# custom-project-specific. _build_demo_candidate_pool() builds that
# pool straight from score_matrix.csv + the real roster, giving every
# premade project the same explanation/runner-up machinery custom
# projects already have, without a second code path.

@st.cache_data
def _build_demo_candidate_pool(project_id: str) -> dict:
    sub = score_matrix[score_matrix.project_id == project_id]
    pool = {}
    for role in sub["role"].unique():
        role_rows = sub[sub.role == role]
        # score_matrix's "role" column is the required slot (all rows
        # here share one value); employees' "role" column is each
        # candidate's own job title. retrieve_adhoc() expects the
        # latter (matches match_all_roles_adhoc()'s shape) -- suffix
        # the collision and drop the required-slot copy.
        merged = role_rows.merge(
            employees[["name", "role", "experience_years",
                       "availability_pct", "skills"]],
            left_on="employee_id", right_index=True, how="left",
            suffixes=("_required", ""),
        )
        merged = merged.drop(columns=["role_required"], errors="ignore")
        pool[role] = merged.sort_values(
            "final_score", ascending=False
        ).reset_index(drop=True)
    return pool


def render_demo_project(project_id: str):
    """
    Renders one Demo-Mode-staffed premade project -- project details
    card, assigned team cards, per-role explanation/runner-up panels,
    full candidate pool -- same visual language as Browse Projects
    mode. Reads assignments from st.session_state.demo_staffed
    (session-only), not staffing_plan.csv.
    """
    proj = all_projects_df.loc[project_id] \
        if project_id in all_projects_df.index else pd.Series()
    assignments_rows = st.session_state.demo_staffed.get(project_id, [])
    project_plan = pd.DataFrame(assignments_rows)

    st.markdown(f"### 📋 {proj.get('project_name', project_id)}")
    st.caption(f"Project ID: {project_id} (Demo Mode, session only)")

    roles_chips = "".join(
        f'<span class="sc-chip">{r}</span>'
        for r in str(proj.get("required_roles", "")).split(";") if r
    )
    skills_chips = "".join(
        f'<span class="sc-chip">{s}</span>'
        for s in str(proj.get("required_skills", "")).split(";") if s
    )
    card_html = (
        '<div class="sc-card">'
        f'<div class="sc-card-sub">Client: {proj.get("client", "—")}</div>'
        '<div style="margin-bottom:10px;">'
        f'{priority_badge(proj.get("priority", "—"))}'
        f'{budget_badge(proj.get("budget_band", "—"))}'
        '</div>'
        '<div class="sc-row">'
        f'<span>⏱ <b>{proj.get("deadline_days", "—")}</b> days deadline</span>'
        f'<span>🎓 <b>{proj.get("min_experience", "—")}</b> yrs min. experience</span>'
        '</div>'
    )
    if roles_chips:
        card_html += f'<div style="margin-top:10px;"><b>Roles:</b> {roles_chips}</div>'
    if skills_chips:
        card_html += f'<div style="margin-top:6px;"><b>Skills:</b> {skills_chips}</div>'
    card_html += '</div>'
    st.markdown(card_html, unsafe_allow_html=True)

    st.markdown("#### Assigned Team")
    if project_plan.empty:
        st.caption(
            "No assignments -- this shouldn't happen for a staffed "
            "demo project; try ending and re-running it."
        )
        return

    display_rows = []
    for _, row in project_plan.iterrows():
        emp_id = row["employee_id"]
        emp = all_employees_df.loc[emp_id] \
            if emp_id in all_employees_df.index else pd.Series()
        display_rows.append({
            "Role": row["role"],
            "Employee": emp.get("name", emp_id),
            "Job Title": emp.get("role", ""),
            "Experience (yr)": emp.get("experience_years", ""),
            "Availability (%)": emp.get("availability_pct", ""),
            "Match Score": f"{row['final_score']:.4f}",
        })

    n_cols = 3
    team_cols = st.columns(n_cols)
    for i, row in enumerate(display_rows):
        with team_cols[i % n_cols]:
            st.markdown(
                '<div class="sc-card">'
                f'<div class="sc-card-title">{row["Employee"]}</div>'
                f'<div class="sc-card-sub">{row["Role"]} · '
                f'{row["Job Title"]}</div>'
                '<div class="sc-row">'
                f'<span>📈 {row["Experience (yr)"]} yrs</span>'
                f'<span>🕒 {row["Availability (%)"]}% avail.</span>'
                f'<span>🎯 {row["Match Score"]}</span>'
                '</div></div>',
                unsafe_allow_html=True,
            )

    st.markdown("#### Role Details & Explanations")

    candidate_pool = _build_demo_candidate_pool(project_id)
    demo_project_dict = {
        **proj.to_dict(),
        "assignments": {r["role"]: r["employee_id"] for r in assignments_rows},
    }

    for _, row in project_plan.iterrows():
        role = row["role"]
        emp_id = row["employee_id"]
        emp = all_employees_df.loc[emp_id] \
            if emp_id in all_employees_df.index else pd.Series()
        score_label = f"{row['final_score']:.4f}"

        ctx = retriever.retrieve_adhoc(
            project_id, role, demo_project_dict, candidate_pool
        )

        with st.expander(
            f"**{role}** → {emp.get('name', emp_id)}  (score: {score_label})"
        ):
            c1, c2, c3 = st.columns(3)
            c1.metric("Experience", f"{emp.get('experience_years', '?')} yrs")
            c2.metric("Availability", f"{emp.get('availability_pct', '?')}%")
            c3.metric("Match Score", score_label)

            if emp.get("skills"):
                st.markdown(f"**Skills:** {emp.get('skills')}")

            if ctx.get("error"):
                st.caption(ctx["error"])
                continue

            st.markdown("---")
            st.markdown("#### 💡 Why was this person selected?")

            session_key = f"demo_explain_{project_id}_{role}"
            if session_key not in st.session_state:
                st.session_state[session_key] = None

            if st.button("Generate Explanation",
                         key=f"demo_btn_{project_id}_{role}"):
                with st.spinner("Generating explanation..."):
                    st.session_state[session_key] = generate_explanation(ctx)

            if st.session_state[session_key]:
                st.info(st.session_state[session_key])

            st.markdown("---")
            st.markdown("#### 🔍 Why not the runner-up?")

            ru = ctx.get("runner_up")
            if ru:
                col_a, col_b = st.columns(2)
                with col_a:
                    st.markdown(f"**✅ Chosen: {emp.get('name', emp_id)}**")
                    st.markdown(f"- Score: `{ctx['assigned']['score']:.4f}`")
                    st.markdown(
                        f"- Experience: {emp.get('experience_years', '?')} yrs")
                    st.markdown(
                        f"- Availability: {emp.get('availability_pct', '?')}%")
                with col_b:
                    direction = "higher" if ru["score_gap"] > 0 else "lower"
                    st.markdown(f"**❌ Runner-up: {ru['name']}**")
                    st.markdown(
                        f"- Score: `{ru['score']:.4f}` ({direction} by "
                        f"`{abs(ru['score_gap']):.4f}`)"
                    )
                    st.markdown(f"- Experience: {ru['experience_years']} yrs")
                    st.markdown(f"- Availability: {ru['availability_pct']}%")

                if ru["score_gap"] > 0:
                    st.warning(
                        f"⚠️ {ru['name']} scored higher for this specific "
                        f"role ({ru['score']:.4f} vs "
                        f"{ctx['assigned']['score']:.4f}) but was assigned "
                        f"elsewhere by the optimizer to maximize the "
                        f"overall team score across all selected projects."
                    )
                else:
                    st.success(
                        f"✅ {emp.get('name', emp_id)} outscored "
                        f"{ru['name']} by {abs(ru['score_gap']):.4f} "
                        f"points for this role."
                    )
            else:
                st.caption("No runner-up data available.")

    st.markdown("#### Full Candidate Pool")
    st.caption(
        "Top 10 eligible candidates per role, ranked by match score "
        "(from score_matrix.csv, the same offline matrix Browse "
        "Projects mode reads for premade projects)."
    )
    for role, role_df in candidate_pool.items():
        assigned_id = demo_project_dict["assignments"].get(role)
        pool = role_df[role_df["eligible"] == True] \
            .head(10).reset_index(drop=True)  # noqa: E712
        pool["rank"] = range(1, len(pool) + 1)
        pool["assigned"] = pool["employee_id"].apply(
            lambda eid: "✅" if eid == assigned_id else ""
        )
        with st.expander(f"**{role}** — top 10 candidates"):
            st.dataframe(
                pool[["rank", "assigned", "name",
                      "employee_id", "final_score"]],
                width="stretch",
                hide_index=True,
            )

# ── Mode: Create Project ─────────────────────────────────────────────
#
# Wires together Phase 3's whole pipeline: save_project() (item 9) ->
# get_capacity_summary() (item 12) shown as a pre-check -> loads the
# Matcher once (cached) -> staff_custom_project() (item 11, which
# internally calls get_busy_employee_ids() from item 10) -> displays
# the result.
#
# On success, this forces a rerun into Browse Projects mode with the
# new project pre-selected. On failure (a role couldn't be staffed),
# it deliberately does NOT rerun or switch modes -- you stay right
# here with the capacity check and error visible, so you can see why
# and adjust the form without losing context. (Per product decision:
# capacity pre-check is a warning, not a hard block on submission --
# the failure message after a failed solve already explains why.)

if mode == "Create Project":
    st.title("➕ Create New Project")
    st.caption(
        "Stage a new project and auto-staff it from **your own saved "
        "employees only** -- not the built-in demo roster (that stays "
        "reserved for 🎬 Demo Mode). Add employees first if your list "
        "is empty."
    )

    # v3 UI update: custom projects match against own_employees_df,
    # never all_employees_df / the demo CSV. An EMPTY plan (not the
    # real staffing_plan.csv) is used for the busy-exclusion check
    # below on purpose -- staffing_plan.csv's E0xx assignments belong
    # to the separate, session-only Demo Mode world. Once an employee
    # id can also exist as one of THIS user's own saved employees
    # (e.g. after running scripts/migrate_demo_roster_to_user.py),
    # treating a Demo Mode assignment as "busy" here would incorrectly
    # shrink this user's own candidate pool over an id collision that
    # has nothing to do with their custom projects. Only this user's
    # OWN custom-project assignments (walked by get_busy_employee_ids()
    # regardless of the plan passed in) should count as "busy" here.
    _no_demo_plan = plan.iloc[0:0]

    if own_employees_df.empty:
        st.warning(
            "👥 You don't have any employees saved yet, so there's "
            "nothing to staff a project with. Head to **Add Employee** "
            "to add people one at a time, or run "
            "`scripts/migrate_demo_roster_to_user.py` once to import "
            "the 80-person demo roster into your own account."
        )
    else:
        with st.expander(
            f"👀 Preview available candidates ({len(own_employees_df)} "
            f"total)", expanded=False,
        ):
            busy_now = get_busy_employee_ids(_no_demo_plan)
            preview = own_employees_df.reset_index()
            preview["Status"] = preview["employee_id"].apply(
                lambda eid: "🔴 Busy" if eid in busy_now else "🟢 Available"
            )
            role_pick = st.multiselect(
                "Filter by role", sorted(own_employees_df["role"].dropna().unique()),
                key="create_project_candidate_filter",
            )
            if role_pick:
                preview = preview[preview["role"].isin(role_pick)]
            st.dataframe(
                preview[["employee_id", "name", "role", "experience_years",
                         "availability_pct", "skills", "Status"]],
                hide_index=True, width="stretch",
            )

    role_options  = sorted(own_employees_df["role"].dropna().unique())
    skill_options = sorted(
        pd.read_csv(BASE / "skills_taxonomy.csv")["skill_name"].unique()
    )

    with st.form("create_project_form"):
        c1, c2 = st.columns(2)
        f_name   = c1.text_input("Project Name")
        f_client = c2.text_input("Client", value="Internal")

        c3, c4, c5 = st.columns(3)
        f_priority = c3.selectbox(
            "Priority", ["low", "medium", "high", "critical"], index=1
        )
        f_min_exp = c4.number_input(
            "Min. Experience (yrs)", min_value=0, max_value=20, value=2
        )
        f_deadline = c5.number_input(
            "Deadline (days)", min_value=1, max_value=365, value=30
        )

        f_budget = st.selectbox("Budget Band", ["low", "medium", "high"], index=1)
        f_roles  = st.multiselect("Required Roles", role_options)
        f_skills = st.multiselect("Required Skills", skill_options)

        submitted = st.form_submit_button(
            "Create & Staff Project", disabled=own_employees_df.empty
        )

    if submitted:
        if not f_name or not f_roles or not f_skills:
            st.error(
                "Project name, at least one required role, and at "
                "least one required skill are all required."
            )
        else:
            record = {
                "project_name":     f_name,
                "client":           f_client or "Internal",
                "priority":         f_priority,
                "min_experience":   int(f_min_exp),
                "deadline_days":    int(f_deadline),
                "budget_band":      f_budget,
                "required_roles":   ";".join(f_roles),
                "required_skills":  ";".join(f_skills),
            }
            pid = save_project(record)
            project = get_project_by_id(pid)
            st.success(f"Project **{pid}** created: {f_name}")

            # Capacity pre-check (item 12) -- cheap, no model call.
            # Shown before the real matcher/solver run so a role with
            # zero available candidates is visible immediately, not
            # just inferred from a failed solve afterward. Per product
            # decision, this warns but does not block submission --
            # the solve below is still attempted.
            cap = get_capacity_summary(project, own_employees_df, _no_demo_plan)
            st.markdown("**Capacity check** (busy employees excluded, "
                        "before running the matcher):")
            cap_rows = [
                {"Role": r, "Available": v["available"],
                 "Total in role": v["in_role"]}
                for r, v in cap["by_role"].items()
            ]
            st.dataframe(pd.DataFrame(cap_rows), hide_index=True)

            zero_roles = [r for r, v in cap["by_role"].items()
                          if v["available"] == 0]
            if zero_roles:
                st.warning(
                    f"⚠️ No available candidates for: "
                    f"{', '.join(zero_roles)}. Staffing will fail for "
                    f"these roles below -- expected, not a bug."
                )

            matcher = load_matcher()
            with st.spinner("Matching and solving..."):
                result = staff_custom_project(
                    project, matcher, _no_demo_plan,
                    employees_df=own_employees_df, verbose=False
                )

            if result.empty:
                st.error(
                    "❌ Could not staff this project -- at least one "
                    "role had zero eligible candidates (check "
                    "min_experience/availability against the capacity "
                    "check above) or the solve was infeasible."
                )
            else:
                st.success("✅ Project staffed")
                display = result.copy()
                display["name"] = display["employee_id"].apply(
                    lambda eid: own_employees_df.loc[eid, "name"]
                    if eid in own_employees_df.index else eid
                )
                st.dataframe(
                    display[["role", "name", "employee_id", "final_score"]],
                    hide_index=True,
                )
                st.caption(
                    f"Project ID: {pid}. Jumping you to its full "
                    f"project view..."
                )

                # Force a fresh top-to-bottom rerun into Browse
                # Projects mode with the new project pre-selected.
                # Necessary, not cosmetic: the sidebar block above
                # already executed -- in Create Project mode, not
                # Browse Projects mode -- before save_project() ran a
                # few lines later in this same pass, so neither
                # "app_mode" nor "project_selector" reflect the new
                # project yet. Stash both under "_pending_*" keys; the
                # top of the next run promotes them before those
                # widgets are instantiated, where it's legal.
                st.session_state["_pending_project_select"] = pid
                st.session_state["_pending_mode"] = "Browse Projects"
                st.rerun()

# ── Mode: 👥 My Employees ─────────────────────────────────────────────
#
# v3 UI addition. Lists ONLY this user's own saved employees
# (own_employees_df / load_own_employees()) -- for
# yashk3404@gmail.com after running
# scripts/migrate_demo_roster_to_user.py, that's the full former demo
# roster too, now owned data instead of a permanently shared CSV.
# Search/role-filter first, then a table, then a "manage one employee"
# panel below with the delete action -- flat "delete button per row"
# doesn't scale once this list is 80+ people.

if mode == "👥 My Employees":
    st.title("👥 My Employees")
    st.caption(
        "Everyone you've saved to your own account. This -- and only "
        "this -- is the candidate pool your custom projects (Create "
        "Project) staff from."
    )

    if own_employees_df.empty:
        st.info(
            "No employees saved yet. Add one via **Add Employee**, or "
            "run `scripts/migrate_demo_roster_to_user.py` once to "
            "import the built-in 80-person demo roster into your own "
            "account in one go."
        )
    else:
        busy_ids = get_busy_employee_ids(plan.iloc[0:0])

        f1, f2 = st.columns([2, 1])
        search_q = f1.text_input(
            "Search by name or employee ID", placeholder="e.g. Aryan or CE001",
        )
        role_filter = f2.selectbox(
            "Filter by role",
            ["All roles"] + sorted(own_employees_df["role"].dropna().unique()),
        )

        filtered = own_employees_df.copy()
        if search_q:
            q = search_q.strip().lower()
            filtered = filtered[
                filtered["name"].astype(str).str.lower().str.contains(q, na=False)
                | filtered.index.astype(str).str.lower().str.contains(q)
            ]
        if role_filter != "All roles":
            filtered = filtered[filtered["role"] == role_filter]

        display = filtered.reset_index()
        display["Status"] = display["employee_id"].apply(
            lambda eid: "🔴 Assigned" if eid in busy_ids else "🟢 Available"
        )
        st.dataframe(
            display[["employee_id", "name", "role", "experience_years",
                     "availability_pct", "department", "location", "Status"]]
            .rename(columns={
                "employee_id": "ID", "name": "Name", "role": "Role",
                "experience_years": "Exp (yrs)",
                "availability_pct": "Availability %",
                "department": "Department", "location": "Location",
            }),
            hide_index=True, width="stretch",
        )
        st.caption(
            f"Showing {len(filtered)} of {len(own_employees_df)} employee(s)."
        )

        st.markdown("---")
        st.subheader("Manage an employee")

        if filtered.empty:
            st.caption("No employees match your search/filter above.")
        else:
            manage_id = st.selectbox(
                "Select employee",
                filtered.index.tolist(),
                format_func=lambda eid: f"{eid} — {filtered.loc[eid, 'name']}",
                key="manage_employee_select",
            )
            emp = filtered.loc[manage_id]
            is_busy = manage_id in busy_ids

            st.markdown(
                f'<div class="sc-card">'
                f'<div class="sc-card-title">{emp.get("name", manage_id)}</div>'
                f'<div class="sc-card-sub">{manage_id} · {emp.get("role", "—")}'
                + (' · <span style="color:#F16A6A">Currently assigned</span>'
                   if is_busy else "")
                + f'</div>'
                f'<div class="sc-row">'
                f'<span>💼 {emp.get("experience_years", "?")} yrs experience</span>'
                f'<span>📈 {emp.get("availability_pct", "?")}% available</span>'
                f'<span>🏢 {emp.get("department") or "—"}</span>'
                f'<span>📍 {emp.get("location") or "—"}</span>'
                f'</div>'
                + (
                    "".join(
                        f'<div class="sc-chip">{s.strip()}</div>'
                        for s in str(emp.get("skills") or "").split(";")
                        if s.strip()
                    )
                )
                + f'</div>',
                unsafe_allow_html=True,
            )

            if st.button("🗑️ Delete Employee", key=f"del_{manage_id}"):
                _confirm_delete_employee(
                    manage_id, emp.get("name", manage_id), is_busy
                )

# ── Mode: Add Employee (items 15 + 16) ───────────────────────────────
#
# Two intake tabs (CV upload / manual entry) both converge on one
# shared review form below -- CV upload populates
# st.session_state["employee_draft"] via parse_resume(), manual entry
# just clears it to {} so the same form renders with blank defaults.
# Neither tab writes to storage directly; only the confirm button at
# the bottom of the review step ever calls save_employee().
#
# Item 17 (merging custom employees into premade candidate pools /
# the matcher's solve pool) is done as of this pass -- see
# all_employees_df above and the employees_df= plumbing through
# Matcher.match()/match_adhoc() and staff_custom_project(). One piece
# is deliberately still out of scope: premade projects' "Full
# Candidate Pool" table (below, in Browse Projects mode) reads the
# offline score_matrix.csv, which is generated against the premade
# roster only -- making a custom employee appear there too would mean
# replacing that static read with a live matcher.match_all_roles()
# call per premade project, a bigger architectural change than "swap
# the call site," and not what this item's plan describes.

if mode == "Add Employee":
    st.title("👤 Add Employee")
    st.caption(
        "Add a candidate via resume upload or a manual form. Both "
        "paths land on the same review step below before anything "
        "is saved -- nothing commits to the roster until you "
        "confirm there."
    )

    tab_cv, tab_manual = st.tabs(["📄 Upload CV", "✏️ Manual Entry"])

    with tab_cv:
        uploaded = st.file_uploader(
            "Resume file", type=["pdf", "docx", "txt"], key="cv_uploader"
        )
        if st.button("Parse Resume", disabled=uploaded is None):
            suffix = Path(uploaded.name).suffix
            with tempfile.NamedTemporaryFile(
                delete=False, suffix=suffix
            ) as tmp:
                tmp.write(uploaded.getvalue())
                tmp_path = tmp.name

            with st.spinner("Extracting and structuring resume..."):
                parsed = parse_resume(tmp_path)

            try:
                os.remove(tmp_path)
            except OSError:
                pass  # best-effort cleanup, not worth failing over

            if parsed.get("error"):
                st.error(f"Could not parse resume: {parsed['error']}")
            else:
                st.session_state["employee_draft"] = {
                    "name":             parsed.get("name") or "",
                    "role":             parsed.get("role") or "",
                    "experience_years": parsed.get("experience_years") or 0,
                    "skills":           parsed.get("skills") or [],
                    "department":       parsed.get("department") or "",
                    "location":         parsed.get("location") or "",
                }
                st.success(
                    f"Parsed via **{parsed['backend']}**. Review and "
                    f"edit below before saving -- nothing's committed yet."
                )
                unrecognized = parsed.get(
                    "skill_validation", {}
                ).get("unrecognized", [])
                if unrecognized:
                    st.caption(
                        f"Not recognized against the skills taxonomy "
                        f"as-extracted (still available to pick "
                        f"manually below): {', '.join(unrecognized)}"
                    )

    with tab_manual:
        st.caption(
            "Starts the review form below with blank fields instead "
            "of resume-extracted ones."
        )
        if st.button("Start Manual Entry"):
            st.session_state["employee_draft"] = {}

    st.markdown("---")
    st.subheader("Review & Confirm")

    if "employee_draft" not in st.session_state:
        st.caption(
            "Upload and parse a resume above, or click **Start "
            "Manual Entry**, to begin."
        )
    else:
        draft = st.session_state["employee_draft"]

        # Real roster role values, not resume_parser.py's hardcoded
        # VALID_ROLES -- that list is explicitly flagged in its own
        # docstring as an unverified assumption, and this dropdown
        # must only ever offer values the matcher actually recognizes.
        role_options = sorted(all_employees_df["role"].dropna().unique())
        skill_options = sorted(
            pd.read_csv(BASE / "skills_taxonomy.csv")["skill_name"].unique()
        )
        skill_lower_to_display = {s.lower(): s for s in skill_options}

        # Pre-check skills via Phase 2 item 8's suggest_skill_matches()
        # -- only exact/fuzzy confidence get pre-checked; "none"
        # confidence suggestions are left unchecked and surfaced as
        # plain text instead, since a wrong guess here is worse than
        # leaving it to a human.
        draft_skills = draft.get("skills") or []
        skill_suggestions = (
            suggest_skill_matches(draft_skills) if draft_skills else []
        )
        suggested_display_skills = sorted({
            skill_lower_to_display[s["suggested_taxonomy_skill"]]
            for s in skill_suggestions
            if s["suggested_taxonomy_skill"] in skill_lower_to_display
        })
        unmatched_skills = [
            s["extracted"] for s in skill_suggestions
            if s["suggested_taxonomy_skill"] is None
        ]

        # Pre-select the role dropdown via suggest_role(), passing the
        # real role_options as valid_roles so any suggestion is
        # guaranteed to actually be a selectable option.
        role_guess = suggest_role(draft.get("role"), valid_roles=role_options)
        suggested_role = role_guess["suggested_role"]
        role_index = (
            role_options.index(suggested_role)
            if suggested_role in role_options else 0
        )

        with st.form("employee_review_form"):
            c1, c2 = st.columns(2)
            f_name = c1.text_input("Name", value=draft.get("name", "") or "")
            f_role = c2.selectbox("Role", role_options, index=role_index)

            c3, c4 = st.columns(2)
            f_exp = c3.number_input(
                "Experience (yrs)", min_value=0, max_value=50,
                value=int(draft.get("experience_years") or 0),
            )
            f_avail = c4.number_input(
                "Availability (%)", min_value=0, max_value=100, value=100,
                help=(
                    "Not extracted from resumes -- this is a business "
                    "input, set it here."
                ),
            )

            c5, c6 = st.columns(2)
            f_dept = c5.text_input(
                "Department", value=draft.get("department", "") or ""
            )
            f_loc = c6.text_input(
                "Location", value=draft.get("location", "") or ""
            )

            f_skills = st.multiselect(
                "Skills", skill_options, default=suggested_display_skills,
            )
            if unmatched_skills:
                st.caption(
                    f"Extracted but not in the taxonomy (not "
                    f"pre-checked above -- add the closest real skill "
                    f"manually if relevant): {', '.join(unmatched_skills)}"
                )

            review_submitted = st.form_submit_button("Review Candidate")

        if review_submitted:
            if not f_name or not f_role:
                st.error("Name and role are required.")
            else:
                candidate = {
                    "name":             f_name,
                    "role":             f_role,
                    "experience_years": int(f_exp),
                    "availability_pct": int(f_avail),
                    # Skills-format fix: the real roster stores skills
                    # as a semicolon-separated string ("Python;AWS;
                    # Docker"), same as required_skills on projects --
                    # not the raw list resume_parser.py extracts or
                    # the multiselect returns. save_employee()'s
                    # record must match employees_with_index.csv's
                    # actual column format, or downstream code that
                    # splits on ";" (candidate pool displays, matcher
                    # input) would silently misread it.
                    "skills":           ";".join(f_skills),
                    "department":       f_dept,
                    "location":         f_loc,
                }
                st.session_state["employee_candidate"] = candidate
                st.session_state["employee_dup_matches"] = (
                    find_possible_duplicates(
                        candidate, employees, list_custom_employees()
                    )
                )

        if "employee_candidate" in st.session_state:
            candidate   = st.session_state["employee_candidate"]
            dup_matches = st.session_state.get("employee_dup_matches", [])

            st.markdown("---")
            if dup_matches:
                warning_lines = "\n\n".join(
                    f"- {line}" for line in format_duplicate_warning(dup_matches)
                )
                st.warning(
                    f"⚠️ Possible duplicate(s) found -- review before "
                    f"confirming:\n\n{warning_lines}"
                )
            else:
                st.success(
                    "✅ No potential duplicates found against the "
                    "current roster."
                )

            bcol1, bcol2 = st.columns(2)
            confirm_label = "➕ Add Anyway" if dup_matches else "➕ Add Employee"

            if bcol1.button(confirm_label, key="confirm_add_employee"):
                emp_id = save_employee(candidate)
                st.success(f"Employee **{emp_id}** added: {candidate['name']}")
                st.caption(
                    "Now eligible everywhere: shows up in the "
                    "Assigned Team / Role Details views, counts "
                    "toward the capacity check, and is a candidate "
                    "for the solver on custom projects. The one "
                    "exception is premade projects' \"Full Candidate "
                    "Pool\" table below, which still reads the "
                    "offline score_matrix.csv (premade employees "
                    "only) rather than live-matching -- a separate, "
                    "bigger change."
                )
                st.session_state.pop("employee_candidate", None)
                st.session_state.pop("employee_dup_matches", None)
                st.session_state.pop("employee_draft", None)

            if bcol2.button("✏️ Discard & Start Over", key="discard_employee"):
                st.session_state.pop("employee_candidate", None)
                st.session_state.pop("employee_dup_matches", None)
                st.session_state.pop("employee_draft", None)
                st.rerun()

# ── Mode: 🎬 Demo Mode ────────────────────────────────────────────
#
# Replaces item 25's "Simulate" mode entirely (product decision --
# see v3 planning). Same underlying mechanism (StaffingOptimizer
# against score_matrix.csv, session-only results, never touches
# staffing_plan.csv or any file on disk) but generalized from the 6
# curated contentious projects to all 30 premade projects, picked on
# demand instead of pre-curated, plus a real "End Project" action
# (un-staff, free the employees) that Simulate never had -- Simulate
# only ever grew its busy-set, it couldn't shrink it.
#
# Deliberately still scoped to the premade pool only, same reasoning
# as old Simulate: StaffingOptimizer reads score_matrix.csv directly,
# a different mechanism from the live ad-hoc matching custom (C0xx)
# projects use. Mixing the two is future scope.
#
# demo_staffed: {project_id: [{"role","employee_id","final_score"}, ...]}
# is the single source of truth here -- who's "busy" in Demo Mode is
# derived from it fresh every render (demo_busy_ids() below), so
# End Project freeing someone is just deleting their project's entry,
# not bookkeeping a separate ever-growing busy set.

if mode == "🎬 Demo Mode":

    ALL_DEMO_PROJECT_IDS = sorted(projects_df.index.tolist())

    st.title("🎬 Demo Mode")
    st.caption(
        "Live-solve any of the 30 premade projects against the demo "
        "roster and see the assigned team, explanations, and runner-up "
        "comparisons -- same engine as Create Project, just against the "
        "built-in dataset instead of your own. Everything here is "
        "session-only; **End Project** un-staffs a project and frees its "
        "people for another run."
    )

    if "demo_staffed" not in st.session_state:
        st.session_state.demo_staffed = {}

    def new_optimizer():
        # Fresh CP-SAT model per call is required -- a solved model
        # can't be rebuilt/re-solved -- so this is a plain factory,
        # not cached; instantiating it is just two pd.read_csv() calls
        # on small local files, cheap enough per click.
        return StaffingOptimizer(
            score_matrix_path=str(BASE / "score_matrix.csv"),
            employees_path=str(BASE / "employees_with_index.csv"),
        )

    def demo_busy_ids() -> set:
        busy = set()
        for rows in st.session_state.demo_staffed.values():
            busy.update(r["employee_id"] for r in rows)
        return busy

    def demo_project_label(pid: str) -> str:
        name = all_projects_df.loc[pid, "project_name"] \
            if pid in all_projects_df.index else pid
        return f"{pid} — {name}"

    runnable_ids = [pid for pid in ALL_DEMO_PROJECT_IDS
                     if pid not in st.session_state.demo_staffed]

    # ── Run projects ─────────────────────────────────────────────
    st.subheader("Run projects")

    if not runnable_ids:
        st.success(
            "All 30 premade projects are currently staffed in Demo "
            "Mode. End one below to free it up for another run."
        )
    else:
        picked = st.multiselect(
            "Select project(s) to solve -- one or many at once",
            runnable_ids,
            format_func=demo_project_label,
            key="demo_pick",
        )

        if st.button("▶️ Run Selected", disabled=not picked):
            busy_now = get_busy_employee_ids(plan) | demo_busy_ids()
            with st.spinner(
                f"Solving {len(picked)} project(s), excluding "
                f"{len(busy_now)} already-busy employee(s)..."
            ):
                opt = new_optimizer()
                opt.build(projects_to_staff=picked, exclude_ids=busy_now)
                result = opt.solve(time_limit_sec=30)

            if result.empty:
                # Same honest all-or-nothing behavior old Simulate had --
                # CP-SAT fails the whole batch the moment one role-slot
                # has zero candidates left, even if other projects in
                # this same selection would otherwise have solved fine.
                if opt.unstaffable_by_exclusion:
                    slots = ", ".join(f"{pid}/{role}" for pid, role
                                       in opt.unstaffable_by_exclusion)
                    st.error(
                        f"Could not staff this selection -- zero "
                        f"eligible candidates left for: {slots}. Try "
                        f"fewer projects at once, or End an "
                        f"already-staffed demo project to free up "
                        f"people."
                    )
                else:
                    st.error(
                        "No feasible assignment for the selected "
                        "project(s) -- try a different combination."
                    )
            else:
                for pid in picked:
                    rows = result[result.project_id == pid]
                    if not rows.empty:
                        st.session_state.demo_staffed[pid] = [
                            {
                                "role":        r["role"],
                                "employee_id": r["employee_id"],
                                "final_score": float(r["final_score"]),
                            }
                            for _, r in rows.iterrows()
                        ]
                st.success(
                    f"Staffed {len(result)} role(s) across "
                    f"{result['project_id'].nunique()} project(s)."
                )
            st.rerun()

    st.markdown("---")

    # ── View / manage staffed demo projects ─────────────────────────
    st.subheader("Staffed demo projects")

    if not st.session_state.demo_staffed:
        st.caption("Nothing staffed yet -- pick project(s) above and run them.")
    else:
        staffed_ids = sorted(st.session_state.demo_staffed.keys())

        top_l, top_r = st.columns([3, 1])
        with top_l:
            view_pid = st.selectbox(
                "View project",
                staffed_ids,
                format_func=demo_project_label,
                key="demo_view_select",
            )
        with top_r:
            st.markdown("<div style='height:28px'></div>",
                        unsafe_allow_html=True)
            if st.button(f"🛑 End {view_pid}", key=f"demo_end_{view_pid}"):
                del st.session_state.demo_staffed[view_pid]
                st.rerun()

        st.caption(
            f"{len(staffed_ids)} project(s) currently staffed in Demo "
            f"Mode, holding {len(demo_busy_ids())} employee(s)."
        )
        st.markdown("---")

        render_demo_project(view_pid)

# ── Mode: Browse Projects ────────────────────────────────────────────

if mode == "Browse Projects":

    proj = all_projects_df.loc[selected_project] \
        if selected_project in all_projects_df.index else pd.Series()

    st.title(f"📋 {proj.get('project_name', selected_project)}")
    st.caption(
        f"Project ID: {selected_project}"
        + (" (custom, this session only)" if is_custom_project else "")
    )

    if is_custom_project:
        # Custom projects never get a staffing_plan.csv row -- their
        # result lives on the project record's "assignments" dict
        # ({role: employee_id}, no final_score -- see project_store.py's
        # update_project_assignments()). Rebuild a plan-shaped frame
        # here so the rest of this section (which all reads
        # project_plan) doesn't need two separate code paths.
        custom_record = get_project_by_id(selected_project)
        assignments = (custom_record or {}).get("assignments", {})

        # v3 UI update: "End Project" now fully removes the project
        # (via delete_project(), not just clearing assignments) and
        # confirms first -- see the _confirm_end_project /
        # _project_ended_popup dialogs defined near the top of this
        # file. Available for any custom project, staffed or not, so
        # an unstaffed one can be cleaned up too, not just a staffed
        # one.
        if st.button(
            f"🛑 End Project"
            + (f" — un-staff {len(assignments)} employee(s) and remove"
               if assignments else " — remove from your project list"),
            key=f"end_custom_{selected_project}",
        ):
            _confirm_end_project(
                selected_project,
                custom_record.get("project_name", selected_project)
                if custom_record else selected_project,
                len(assignments),
            )

        project_plan = pd.DataFrame(
            [
                {
                    "project_id": selected_project,
                    "role": role,
                    "employee_id": emp_id,
                    "final_score": None,
                }
                for role, emp_id in assignments.items()
            ],
            columns=["project_id", "role", "employee_id", "final_score"],
        )
    else:
        project_plan = plan[
            plan.project_id == selected_project
        ].reset_index(drop=True)

    # ── Project details ───────────────────────────────────────────

    st.subheader("Project Details")

    roles_chips = "".join(
        f'<span class="sc-chip">{r}</span>'
        for r in str(proj.get("required_roles", "")).split(";") if r
    )
    skills_chips = "".join(
        f'<span class="sc-chip">{s}</span>'
        for s in str(proj.get("required_skills", "")).split(";") if s
    )

    card_html = (
        '<div class="sc-card">'
        f'<div class="sc-card-sub">Client: {proj.get("client", "—")}</div>'
        '<div style="margin-bottom:10px;">'
        f'{priority_badge(proj.get("priority", "—"))}'
        f'{budget_badge(proj.get("budget_band", "—"))}'
        '</div>'
        '<div class="sc-row">'
        f'<span>⏱ <b>{proj.get("deadline_days", "—")}</b> days deadline</span>'
        f'<span>🎓 <b>{proj.get("min_experience", "—")}</b> yrs min. experience</span>'
        '</div>'
    )
    if roles_chips:
        card_html += f'<div style="margin-top:10px;"><b>Roles:</b> {roles_chips}</div>'
    if skills_chips:
        card_html += f'<div style="margin-top:6px;"><b>Skills:</b> {skills_chips}</div>'
    card_html += '</div>'

    st.markdown(card_html, unsafe_allow_html=True)

    st.markdown("---")

    # ── Overview table ─────────────────────────────────────────────

    st.subheader("Assigned Team")

    if is_custom_project and project_plan.empty:
        st.caption(
            "No assignments on this project yet -- it may have been "
            "created but not yet staffed, or staffing failed for "
            "every role. Go to **Create Project** mode to check."
        )

    display_rows = []
    for _, row in project_plan.iterrows():
        emp_id = row["employee_id"]
        emp    = all_employees_df.loc[emp_id] if emp_id in all_employees_df.index \
                 else pd.Series()
        score  = row["final_score"]
        display_rows.append({
            "Role":             row["role"],
            "Employee":         emp.get("name",             emp_id),
            "Job Title":        emp.get("role",             ""),
            "Experience (yr)":  emp.get("experience_years", ""),
            "Availability (%)": emp.get("availability_pct", ""),
            "Match Score":      f"{score:.4f}" if pd.notna(score) else "—",
        })

    if display_rows:
        n_cols = 3
        team_cols = st.columns(n_cols)
        for i, row in enumerate(display_rows):
            with team_cols[i % n_cols]:
                st.markdown(
                    '<div class="sc-card">'
                    f'<div class="sc-card-title">{row["Employee"]}</div>'
                    f'<div class="sc-card-sub">{row["Role"]} · '
                    f'{row["Job Title"]}</div>'
                    '<div class="sc-row">'
                    f'<span>📈 {row["Experience (yr)"]} yrs</span>'
                    f'<span>🕒 {row["Availability (%)"]}% avail.</span>'
                    f'<span>🎯 {row["Match Score"]}</span>'
                    '</div></div>',
                    unsafe_allow_html=True,
                )

    # ── Per-role detail panels ──────────────────────────────────────

    st.subheader("Role Details & Explanations")

    # Item 23 -- fetched once per project render, not per role: every
    # role in the loop below reads from the same pool.
    # get_candidate_pool() returns None for a custom project that was
    # never solved, or was solved in a session predating item 21 --
    # both real cases, not bugs, so this is handled per-role below
    # rather than assumed away here.
    candidate_pool = get_candidate_pool(selected_project) \
        if is_custom_project else None

    if is_custom_project and not project_plan.empty and candidate_pool is None:
        st.caption(
            "This project was staffed in a session before live "
            "explanations were added (or its candidate pool wasn't "
            "saved) -- re-run the solve from **Create Project** mode "
            "to enable explanations and runner-up comparisons below."
        )

    for _, row in project_plan.iterrows():
        role   = row["role"]
        emp_id = row["employee_id"]
        emp    = all_employees_df.loc[emp_id] if emp_id in all_employees_df.index \
                 else pd.Series()
        score  = row["final_score"]
        score_label = f"{score:.4f}" if pd.notna(score) else "—"

        # Item 23 follow-up -- fetch ctx once per role, up front, and
        # reuse it for the Match Score metric AND the explanation/
        # runner-up sections below (was being fetched twice further
        # down before, and never at all before the stats render).
        # row["final_score"] is always None for custom projects (they
        # have no staffing_plan.csv row), so score_label above stays
        # stuck at "—" for them unless it's overwritten here from
        # ctx["assigned"]["score"], which is the actual solved score
        # sitting in the candidate pool.
        ctx = None
        if is_custom_project:
            if candidate_pool is not None:
                ctx = retriever.retrieve_adhoc(
                    selected_project, role, custom_record, candidate_pool
                )
                if not ctx.get("error"):
                    score_label = f"{ctx['assigned']['score']:.4f}"
        else:
            ctx = retriever.retrieve(selected_project, role)

        with st.expander(
            f"**{role}** → {emp.get('name', emp_id)}  "
            f"(score: {score_label})"
        ):

            # ── Stats ─────────────────────────────────────────────
            c1, c2, c3 = st.columns(3)
            c1.metric("Experience",
                      f"{emp.get('experience_years', '?')} yrs")
            c2.metric("Availability",
                      f"{emp.get('availability_pct', '?')}%")
            c3.metric("Match Score", score_label)

            if emp.get("skills"):
                st.markdown(f"**Skills:** {emp.get('skills')}")

            if is_custom_project and candidate_pool is None:
                # Caption above already explains why -- nothing more
                # to show for this role specifically.
                continue

            st.markdown("---")

            # ── LLM Explanation ──────────────────────────────────
            st.markdown("#### 💡 Why was this person selected?")

            session_key = f"explain_{selected_project}_{role}"
            if session_key not in st.session_state:
                st.session_state[session_key] = None

            if st.button("Generate Explanation",
                         key=f"btn_{selected_project}_{role}"):
                with st.spinner("Generating explanation with Ollama..."):
                    if ctx.get("error"):
                        st.session_state[session_key] = None
                        st.error(ctx["error"])
                    else:
                        st.session_state[session_key] = \
                            generate_explanation(ctx)

            if st.session_state[session_key]:
                st.info(st.session_state[session_key])

            st.markdown("---")

            # ── Why not X? ────────────────────────────────────────
            st.markdown("#### 🔍 Why not the runner-up?")

            if ctx.get("error"):
                st.caption(ctx["error"])
                continue

            ru = ctx.get("runner_up")

            if ru:
                col_a, col_b = st.columns(2)

                with col_a:
                    st.markdown(
                        f"**✅ Chosen: {emp.get('name', emp_id)}**")
                    st.markdown(f"- Score: `{ctx['assigned']['score']:.4f}`")
                    st.markdown(
                        f"- Experience: "
                        f"{emp.get('experience_years', '?')} yrs")
                    st.markdown(
                        f"- Availability: "
                        f"{emp.get('availability_pct', '?')}%")

                with col_b:
                    direction = "higher" if ru["score_gap"] > 0 \
                                else "lower"
                    st.markdown(
                        f"**❌ Runner-up: {ru['name']}**")
                    st.markdown(
                        f"- Score: `{ru['score']:.4f}` "
                        f"({direction} by "
                        f"`{abs(ru['score_gap']):.4f}`)")
                    st.markdown(
                        f"- Experience: {ru['experience_years']} yrs")
                    st.markdown(
                        f"- Availability: {ru['availability_pct']}%")

                if ru["score_gap"] > 0:
                    st.warning(
                        f"⚠️ {ru['name']} scored higher for this "
                        f"specific role ({ru['score']:.4f} vs "
                        f"{ctx['assigned']['score']:.4f}) but was "
                        f"assigned elsewhere by the optimizer to "
                        f"maximize the overall team score across all "
                        f"projects."
                    )
                else:
                    st.success(
                        f"✅ {emp.get('name', emp_id)} outscored "
                        f"{ru['name']} by "
                        f"{abs(ru['score_gap']):.4f} points for "
                        f"this role."
                    )
            else:
                st.caption("No runner-up data available.")

    # ── Candidate pool ──────────────────────────────────────────────

    st.subheader("Full Candidate Pool")

    if is_custom_project:
        # v3 UI update -- previously "not available" here (this used
        # to only read the offline score_matrix.csv, which has no
        # rows for C0xx projects). Now renders the same
        # candidate_pool this project's Role Details section above
        # already fetched via get_candidate_pool() -- the full ranked
        # {role: DataFrame} the solve produced, straight from THIS
        # user's own employee list (own_employees_df), not the demo
        # roster.
        if candidate_pool is None:
            st.caption(
                "No candidate pool saved for this project yet -- "
                "solve it (or re-solve it) from **Create Project** to "
                "see every eligible candidate per role here."
            )
        else:
            st.caption(
                "Every eligible candidate from your own employee "
                "list, ranked by match score for each role. ✅ marks "
                "who was actually assigned."
            )
            for role, role_df in candidate_pool.items():
                assigned_rows = project_plan[project_plan.role == role]
                assigned_id = assigned_rows.iloc[0]["employee_id"] \
                    if not assigned_rows.empty else None

                pool = role_df.sort_values(
                    "final_score", ascending=False
                ).reset_index(drop=True)
                pool["rank"] = range(1, len(pool) + 1)
                pool["assigned"] = pool["employee_id"].apply(
                    lambda eid: "✅" if eid == assigned_id else ""
                )

                cols = [c for c in
                        ["rank", "assigned", "name", "employee_id", "final_score"]
                        if c in pool.columns]
                with st.expander(f"**{role}** — {len(pool)} candidate(s)"):
                    st.dataframe(pool[cols], width="stretch", hide_index=True)
    else:
        st.caption(
            "Top 10 eligible candidates per role, ranked by match "
            "score. Sourced from the offline score_matrix.csv, so "
            "session-added custom employees won't appear here even "
            "though they're now eligible for custom projects' solves."
        )

        for _, row in project_plan.iterrows():
            role        = row["role"]
            assigned_id = project_plan[
                project_plan.role == role
            ].iloc[0]["employee_id"]

            pool = (
                score_matrix[
                    (score_matrix.project_id == selected_project) &
                    (score_matrix.role == role) &
                    (score_matrix.eligible == True)
                ]
                .sort_values("final_score", ascending=False)
                .head(10)
                .reset_index(drop=True)
            )

            pool["rank"] = range(1, len(pool) + 1)
            pool["name"] = pool["employee_id"].apply(
                lambda eid: employees.loc[eid, "name"]
                if eid in employees.index else eid
            )
            pool["assigned"] = pool["employee_id"].apply(
                lambda eid: "✅" if eid == assigned_id else ""
            )

            with st.expander(f"**{role}** — top 10 candidates"):
                st.dataframe(
                    pool[["rank", "assigned", "name",
                          "employee_id", "final_score"]],
                    width="stretch",
                    hide_index=True,
                )

    # ── SHAP Panel (added Day 28) ────────────────────────────────────

    st.markdown("---")
    st.subheader("📊 Feature Importance (SHAP)")
    st.caption(
        "Which factors drive match scores across all candidates. "
        "This panel is global (not project-specific), so it's the same "
        "regardless of which project -- premade or custom -- is "
        "selected above."
    )

    shap_summary    = BASE / "shap_summary.png"
    shap_importance = BASE / "shap_importance.png"

    col_s, col_i = st.columns(2)
    if shap_summary.exists():
        col_s.image(str(shap_summary),
                    caption="SHAP Summary Plot")
    else:
        col_s.caption("Run 13_shap.ipynb to generate this plot.")

    if shap_importance.exists():
        col_i.image(str(shap_importance),
                    caption="Mean Feature Importance")
    else:
        col_i.caption("Run 13_shap.ipynb to generate this plot.")