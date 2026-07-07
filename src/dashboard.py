# src/dashboard.py
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
from retrieve_context import ContextRetriever
from generate_explanation import generate_explanation

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

# ── Sidebar ───────────────────────────────────────────────────────

st.sidebar.title("🧠 Staffing Copilot")
st.sidebar.markdown("---")

project_ids = sorted(plan["project_id"].unique())

def _project_label(pid):
    name = projects_df.loc[pid, "project_name"] \
        if pid in projects_df.index else ""
    return f"{pid} — {name}" if name else pid

selected_project = st.sidebar.selectbox(
    "Select Project",
    project_ids,
    format_func=_project_label,
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    f"**Total projects:** {len(project_ids)}  \n"
    f"**Total assignments:** {len(plan)}  \n"
    f"**Avg match score:** {plan.final_score.mean():.3f}"
)

# ── Main ──────────────────────────────────────────────────────────

proj = projects_df.loc[selected_project] \
    if selected_project in projects_df.index else pd.Series()

st.title(f"📋 {proj.get('project_name', selected_project)}")
st.caption(f"Project ID: {selected_project}")

project_plan = plan[
    plan.project_id == selected_project
].reset_index(drop=True)

# ── Project details ───────────────────────────────────────────────

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

# ── Overview table ────────────────────────────────────────────────

st.subheader("Assigned Team")

display_rows = []
for _, row in project_plan.iterrows():
    emp_id = row["employee_id"]
    emp    = employees.loc[emp_id] if emp_id in employees.index \
             else pd.Series()
    display_rows.append({
        "Role":             row["role"],
        "Employee":         emp.get("name",             emp_id),
        "Job Title":        emp.get("role",             ""),
        "Experience (yr)":  emp.get("experience_years", ""),
        "Availability (%)": emp.get("availability_pct", ""),
        "Match Score":      f"{row['final_score']:.4f}",
    })

st.dataframe(pd.DataFrame(display_rows), width="stretch")

# ── Per-role detail panels ────────────────────────────────────────

st.subheader("Role Details & Explanations")

for _, row in project_plan.iterrows():
    role   = row["role"]
    emp_id = row["employee_id"]
    emp    = employees.loc[emp_id] if emp_id in employees.index \
             else pd.Series()

    with st.expander(
        f"**{role}** → {emp.get('name', emp_id)}  "
        f"(score: {row['final_score']:.4f})"
    ):

        # ── Stats ─────────────────────────────────────────────────
        c1, c2, c3 = st.columns(3)
        c1.metric("Experience",
                  f"{emp.get('experience_years', '?')} yrs")
        c2.metric("Availability",
                  f"{emp.get('availability_pct', '?')}%")
        c3.metric("Match Score",
                  f"{row['final_score']:.4f}")

        if emp.get("skills"):
            st.markdown(f"**Skills:** {emp.get('skills')}")

        st.markdown("---")

        # ── LLM Explanation ───────────────────────────────────────
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

        # ── Why not X? ────────────────────────────────────────────
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

# ── Candidate pool ────────────────────────────────────────────────

st.subheader("Full Candidate Pool")
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

# ── SHAP Panel (added Day 28) ─────────────────────────────────────

st.markdown("---")
st.subheader("📊 Feature Importance (SHAP)")
st.caption(
    "Which factors drive match scores across all candidates. "
    "Run notebooks/13_shap.ipynb to generate these plots."
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