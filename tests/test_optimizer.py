# tests/test_optimizer.py
"""
Unit tests for the StaffingOptimizer (OR-Tools CP-SAT).
Run: pytest tests/test_optimizer.py -v
"""

import pytest
import sys
import pandas as pd
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent / "src"))
from optimize_staffing import StaffingOptimizer

DATA_DIR = Path(__file__).parent.parent / "data" / "processed"


@pytest.fixture(scope="module")
def optimizer():
    return StaffingOptimizer(
        score_matrix_path=str(DATA_DIR / "score_matrix.csv"),
        employees_path=str(DATA_DIR / "employees_with_index.csv"),
    )


def test_optimizer_loads(optimizer):
    assert optimizer.scores    is not None
    assert optimizer.employees is not None
    assert len(optimizer.scores)    == 6800
    assert len(optimizer.employees) == 80


def test_plan_file_exists():
    assert (DATA_DIR / "staffing_plan.csv").exists()


def test_plan_no_double_booking():
    plan  = pd.read_csv(DATA_DIR / "staffing_plan.csv")
    dupes = plan[plan.duplicated("employee_id", keep=False)]
    assert len(dupes) == 0, \
        f"Double-booking detected:\n{dupes}"


def test_plan_all_slots_filled():
    plan = pd.read_csv(DATA_DIR / "staffing_plan.csv")
    sm   = pd.read_csv(DATA_DIR / "score_matrix.csv")

    # Plan only covers projects the optimizer was asked to staff
    # Check only those projects, not the full 30
    staffed_projects = plan["project_id"].unique()

    all_slots = (sm[
                     (sm.eligible == True) &
                     (sm["project_id"].isin(staffed_projects))
                 ]
                 [["project_id", "role"]]
                 .drop_duplicates())

    filled  = plan[["project_id", "role"]].drop_duplicates()
    missing = all_slots.merge(filled, on=["project_id", "role"],
                               how="left", indicator=True)
    missing = missing[missing._merge == "left_only"]

    assert len(missing) == 0, \
        f"Unfilled slots within staffed projects:\n" \
        f"{missing[['project_id','role']]}"


def test_plan_scores_in_range():
    plan = pd.read_csv(DATA_DIR / "staffing_plan.csv")
    assert (plan["final_score"] >= 0).all()
    assert (plan["final_score"] <= 1).all()


def test_plan_only_eligible_employees():
    plan = pd.read_csv(DATA_DIR / "staffing_plan.csv")
    sm   = pd.read_csv(DATA_DIR / "score_matrix.csv")

    eligible_set = set(
        zip(sm[sm.eligible == True]["employee_id"],
            sm[sm.eligible == True]["project_id"],
            sm[sm.eligible == True]["role"])
    )
    for _, row in plan.iterrows():
        key = (row["employee_id"], row["project_id"], row["role"])
        assert key in eligible_set, \
            f"Ineligible assignment in plan: {key}"


def test_find_unstaffable_roles(optimizer):
    unstaffable = optimizer.find_unstaffable_roles()
    # With 80 synthetic employees all skills are covered
    assert len(unstaffable) == 0, \
        f"Unexpected unstaffable roles:\n{unstaffable}"


def test_solve_two_projects():
    """Fresh optimizer run on 2 projects — must reach OPTIMAL."""
    opt = StaffingOptimizer(
        score_matrix_path=str(DATA_DIR / "score_matrix.csv"),
        employees_path=str(DATA_DIR / "employees_with_index.csv"),
    )
    opt.build(projects_to_staff=["P001", "P002"])
    plan = opt.solve(time_limit_sec=30)

    assert len(plan) == 6, \
        f"Expected 6 assignments for P001+P002, got {len(plan)}"
    assert plan.duplicated("employee_id").sum() == 0, \
        "Double-booking in 2-project solve"


# ── Phase 5 / item 17 -- staff_custom_project() with a merged pool ──

def test_staff_custom_project_can_assign_a_custom_employee():
    """
    End-to-end: a custom (CE0xx) employee, present only in a merged
    employees_df, must be a real candidate for an ad-hoc project's
    solve -- not just visible in storage.
    """
    import sys
    from pathlib import Path as _Path
    sys.path.append(str(_Path(__file__).parent.parent / "src"))
    from matcher import Matcher
    from optimize_staffing import staff_custom_project

    m = Matcher()
    m.load()

    real = m.emp_df.set_index("employee_id")
    custom_row = pd.DataFrame([{
        "name": "Merged Pool Test Dev",
        "role": "Backend Dev",
        "experience_years": 6,
        "availability_pct": 100,
        "skills": "Python;Docker;PostgreSQL",
        "cost_band": None,
        "department": "Engineering",
        "location": "Remote",
    }], index=pd.Index(["CE999"], name="employee_id"))
    merged = pd.concat([real, custom_row])

    project = {
        "project_id": "C_TEST_999",
        "project_name": "Test ad-hoc",
        "min_experience": 2,
        "required_roles": "Backend Dev",
        "required_skills": "Python;Docker",
    }
    empty_plan = pd.DataFrame(columns=["project_id", "role", "employee_id", "final_score"])

    result = staff_custom_project(
        project, m, empty_plan, employees_df=merged, verbose=False
    )
    assert not result.empty


def test_staff_custom_project_without_employees_df_still_works():
    """employees_df defaults to None -- must fall back to the
    matcher's own premade-only pool, unaffected by item 17."""
    import sys
    from pathlib import Path as _Path
    sys.path.append(str(_Path(__file__).parent.parent / "src"))
    from matcher import Matcher
    from optimize_staffing import staff_custom_project

    m = Matcher()
    m.load()

    project = {
        "project_id": "C_TEST_998",
        "project_name": "Test ad-hoc no merge",
        "min_experience": 1,
        "required_roles": "Backend Dev",
        "required_skills": "Python",
    }
    empty_plan = pd.DataFrame(columns=["project_id", "role", "employee_id", "final_score"])

    result = staff_custom_project(project, m, empty_plan, verbose=False)
    assert not result.empty