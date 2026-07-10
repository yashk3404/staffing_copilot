"""
Staffing Copilot — Streamlit Dashboard v2
Adds LLM explanation panels and "Why not X?" runner-up view.

Run:
    venv\\Scripts\\python.exe -m streamlit run src/dashboard.py
"""

import os
import sys
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
)
from src.optimize_staffing import staff_custom_project

BASE = Path(__file__).parent.parent / "data" / "processed"

st.set_page_config(
    page_title="Staffing Copilot",
    page_icon="🧠",
    layout="wide",
)

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
    ["Browse Projects", "Create Project", "Add Employee"],
    key="app_mode",
)

st.sidebar.markdown("---")

# ── Item 14 -- unified project selector (premade + custom), reading
# through load_all_projects() (item 9) instead of only projects_df.
#
# Scope note: premade projects only show here if they already have a
# staffing_plan.csv assignment -- same behavior as before item 14,
# not a new restriction. Showing all 30 premade projects (including
# the 25 with no assignments yet) is a bigger UX question -- what
# does an unstaffed premade project's "Assigned Team" table even show
# -- left for a later phase, not part of this wiring.
#
# Custom (C0xx) projects always show once created, since save_project()
# only runs when the user deliberately submits the Create Project form.
all_projects_df = load_all_projects(projects_df)

staffed_premade_ids = set(plan["project_id"].unique())
custom_ids = set(all_projects_df.index) - set(projects_df.index)
project_ids = sorted(staffed_premade_ids | custom_ids)

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
        tag = " (custom)" if pid in custom_ids else ""
        return f"{pid} — {name}{tag}" if name else pid

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

    is_custom_project = selected_project in custom_ids

st.sidebar.markdown("---")
st.sidebar.markdown(
    f"**Total projects:** {len(project_ids)}  \n"
    f"**Total assignments:** {len(plan)}  \n"
    f"**Avg match score:** {plan.final_score.mean():.3f}"
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
        "Stage a new project and auto-staff it against employees not "
        "already committed to a premade or another custom project "
        "this session."
    )

    role_options  = sorted(employees["role"].unique())
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

        submitted = st.form_submit_button("Create & Staff Project")

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
            cap = get_capacity_summary(project, employees, plan)
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
                    project, matcher, plan, verbose=False
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
                    lambda eid: employees.loc[eid, "name"]
                    if eid in employees.index else eid
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

# ── Mode: Add Employee (item 15 -- not built yet) ────────────────────

if mode == "Add Employee":
    st.title("👤 Add Employee")
    st.info(
        "Not built yet -- this is item 15 on the Phase 4 plan: a "
        "tabbed CV-upload / manual-form interface converging on a "
        "mandatory review step (using Phase 2's "
        "suggest_skill_matches() / suggest_role() for pre-checked "
        "suggestions), followed by item 16's duplicate-warning UI "
        "before final commit."
    )

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

    d1, d2, d3, d4 = st.columns(4)
    d1.metric("Client", proj.get("client", "—"))
    d2.metric("Priority", str(proj.get("priority", "—")).title())
    d3.metric("Min. Experience", f"{proj.get('min_experience', '—')} yrs")
    d4.metric("Deadline", f"{proj.get('deadline_days', '—')} days")

    st.markdown(f"**Budget Band:** {str(proj.get('budget_band', '—')).title()}")

    required_roles = proj.get("required_roles", "")
    if required_roles:
        st.markdown(
            "**Required Roles:** "
            + ", ".join(str(required_roles).split(";"))
        )

    required_skills = proj.get("required_skills", "")
    if required_skills:
        st.markdown(
            "**Required Skills:** "
            + ", ".join(str(required_skills).split(";"))
        )

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
        emp    = employees.loc[emp_id] if emp_id in employees.index \
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
        st.dataframe(pd.DataFrame(display_rows), width="stretch")

    # ── Per-role detail panels ──────────────────────────────────────

    st.subheader("Role Details & Explanations")

    if is_custom_project and not project_plan.empty:
        st.caption(
            "LLM explanations and runner-up comparisons below aren't "
            "available for custom-created projects -- that context is "
            "generated offline (RAG index + SHAP values) against the "
            "premade project set only. Stats and skills are still "
            "shown from the live employee roster."
        )

    for _, row in project_plan.iterrows():
        role   = row["role"]
        emp_id = row["employee_id"]
        emp    = employees.loc[emp_id] if emp_id in employees.index \
                 else pd.Series()
        score  = row["final_score"]
        score_label = f"{score:.4f}" if pd.notna(score) else "—"

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

            if is_custom_project:
                # No RAG context or runner-up data exists for custom
                # projects -- retriever.retrieve() is built against
                # the premade project set only (see caption above).
                # Skip straight past the explanation/runner-up
                # sections rather than call it and surface a
                # confusing empty/wrong result.
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
                    ctx = retriever.retrieve(selected_project, role)
                    st.session_state[session_key] = \
                        generate_explanation(ctx)

            if st.session_state[session_key]:
                st.info(st.session_state[session_key])

            st.markdown("---")

            # ── Why not X? ────────────────────────────────────────
            st.markdown("#### 🔍 Why not the runner-up?")

            ctx = retriever.retrieve(selected_project, role)
            ru  = ctx.get("runner_up")

            if ru:
                col_a, col_b = st.columns(2)

                with col_a:
                    st.markdown(
                        f"**✅ Chosen: {emp.get('name', emp_id)}**")
                    st.markdown(f"- Score: `{row['final_score']:.4f}`")
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
                        f"{row['final_score']:.4f}) but was assigned "
                        f"elsewhere by the optimizer to maximize the "
                        f"overall team score across all projects."
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
        st.caption(
            "Not available for custom-created projects -- this table "
            "comes from score_matrix.csv, which is generated offline "
            "in the notebooks against the premade project set only, "
            "and has no rows for session-created C0xx projects."
        )
    else:
        st.caption(
            "Top 10 eligible candidates per role, ranked by match score.")

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

            st.markdown(f"**{role} — top 10:**")
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
        "Run notebooks/13_shap.ipynb to generate these plots. This "
        "panel is global (not project-specific), so it's the same "
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