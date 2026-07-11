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


# ── Item 26 -- exclude_ids (item 24) ────────────────────────────────
#
# Synthetic score data throughout, not the bundled score_matrix.csv --
# it turned out (discovered while building item 24) every role within
# one premade project shares an identical eligible pool, which would
# mask exactly the bug these tests exist to catch. Hand-built data
# isolates one role's candidate pool from another's on purpose.

@pytest.fixture
def synthetic_optimizer():
    """
    Two projects, three roles, deliberately different-sized candidate
    pools per role so exclude_ids can be aimed at exactly one role's
    pool without touching another's -- PX/Backend Dev has 2 eligible
    candidates, PX/Frontend Dev has exactly 1 (E3), PY/Backend Dev has
    its own separate pool (E4) so cross-project behavior is covered
    too.
    """
    opt = StaffingOptimizer.__new__(StaffingOptimizer)
    from ortools.sat.python import cp_model as _cp_model
    opt.model = _cp_model.CpModel()
    opt.x = {}
    opt.solver = _cp_model.CpSolver()
    opt.df = None
    opt.unstaffable_by_exclusion = []
    opt.employees = pd.DataFrame()
    opt.scores = pd.DataFrame([
        {"project_id": "PX", "role": "Backend Dev",  "employee_id": "E1", "final_score": 0.8, "eligible": True},
        {"project_id": "PX", "role": "Backend Dev",  "employee_id": "E2", "final_score": 0.6, "eligible": True},
        {"project_id": "PX", "role": "Frontend Dev", "employee_id": "E3", "final_score": 0.7, "eligible": True},
        {"project_id": "PY", "role": "Backend Dev",  "employee_id": "E4", "final_score": 0.5, "eligible": True},
    ])
    return opt


class TestExcludeIds:
    def test_excluded_rows_dropped_before_solving(self, synthetic_optimizer):
        synthetic_optimizer.build(projects_to_staff=["PX"], exclude_ids={"E1"})
        assert "E1" not in set(synthetic_optimizer.df.employee_id)
        assert "E2" in set(synthetic_optimizer.df.employee_id)

    def test_no_exclude_ids_behaves_exactly_as_before(self, synthetic_optimizer):
        synthetic_optimizer.build(projects_to_staff=["PX"])
        assert set(synthetic_optimizer.df.employee_id) == {"E1", "E2", "E3"}
        assert synthetic_optimizer.unstaffable_by_exclusion == []

    def test_partial_exclusion_still_solves_via_remaining_candidate(self, synthetic_optimizer):
        """E1 excluded, but E2 still covers Backend Dev -- must reassign,
        not fail."""
        synthetic_optimizer.build(projects_to_staff=["PX"], exclude_ids={"E1"})
        result = synthetic_optimizer.solve(time_limit_sec=5)
        assert not result.empty
        assert set(result[result.role == "Backend Dev"].employee_id) == {"E2"}

    def test_full_exclude_of_a_roles_only_candidate_is_infeasible_not_a_crash(
        self, synthetic_optimizer
    ):
        """Frontend Dev's ONLY eligible candidate (E3) excluded --
        must produce a clean INFEASIBLE-shaped empty result, not raise."""
        synthetic_optimizer.build(projects_to_staff=["PX"], exclude_ids={"E3"})
        assert ("PX", "Frontend Dev") in synthetic_optimizer.unstaffable_by_exclusion
        result = synthetic_optimizer.solve(time_limit_sec=5)  # must not raise
        assert result.empty

    def test_full_exclude_fails_whole_batch_not_just_starved_role(self, synthetic_optimizer):
        """PX/Backend Dev (E1,E2 both fine) sits in the SAME batch as
        PX/Frontend Dev (E3 excluded) -- CP-SAT contradiction fails the
        whole model, not just the starved slot. This is deliberate
        (see optimize_staffing.py build()) -- locking it in so a future
        change doesn't accidentally start returning partial results
        without dashboard.py's Simulate mode being updated to match."""
        synthetic_optimizer.build(projects_to_staff=["PX"], exclude_ids={"E3"})
        result = synthetic_optimizer.solve(time_limit_sec=5)
        assert result.empty  # PX/Backend Dev's otherwise-valid assignment is gone too

    def test_exclude_everyone_returns_empty_df_with_columns_not_a_crash(
        self, synthetic_optimizer
    ):
        """Every candidate for every role excluded -> zero decision
        variables at all. Historically raised KeyError from
        pd.DataFrame([]).sort_values(...) -- must return a
        properly-columned empty DataFrame instead."""
        synthetic_optimizer.build(
            projects_to_staff=["PX"], exclude_ids={"E1", "E2", "E3"}
        )
        result = synthetic_optimizer.solve(time_limit_sec=5)  # must not raise
        assert result.empty
        assert list(result.columns) == \
            ["project_id", "role", "employee_id", "final_score"]

    def test_exclusion_in_one_project_does_not_affect_another(self, synthetic_optimizer):
        """Excluding E3 (PX/Frontend Dev's only candidate) must not
        touch PY, which has its own separate pool (E4)."""
        synthetic_optimizer.build(
            projects_to_staff=["PX", "PY"], exclude_ids={"E3"}
        )
        assert synthetic_optimizer.unstaffable_by_exclusion == [("PX", "Frontend Dev")]
        assert "E4" in set(synthetic_optimizer.df.employee_id)


# ── solve_ad_hoc_project() empty-result shape ────────────────────────
#
# Found during a post-Item-26 review: solve_ad_hoc_project() (the
# function staff_custom_project() actually calls for every custom
# project) had the exact bug Item 26 fixed on StaffingOptimizer.solve()
# -- a bare columnless pd.DataFrame() on both its empty-return paths --
# just missed because it's a separate function. Never crashed since
# every caller checks .empty first, but had zero direct test coverage.
# These tests build role_scores by hand, so no sentence_transformers
# dependency needed.

from optimize_staffing import solve_ad_hoc_project  # noqa: E402


class TestSolveAdHocProjectEmptyResult:

    def test_zero_eligible_candidates_returns_columned_empty_df(self):
        """A role with zero eligible candidates must return an empty
        df with the documented columns, not a bare pd.DataFrame()."""
        role_scores = {
            "Backend Dev": pd.DataFrame([
                {"employee_id": "E1", "final_score": 0.9, "eligible": False},
            ]),
        }
        result = solve_ad_hoc_project(role_scores)
        assert result.empty
        assert list(result.columns) == ["role", "employee_id", "final_score"]

    def test_infeasible_solve_returns_columned_empty_df(self):
        """Two roles, one shared candidate, both roles require someone
        -- infeasible (can't fill both with only one eligible person
        overlapping every role). Must return the documented empty
        shape, not a bare pd.DataFrame()."""
        role_scores = {
            "Backend Dev": pd.DataFrame([
                {"employee_id": "E1", "final_score": 0.9, "eligible": True},
            ]),
            "Frontend Dev": pd.DataFrame([
                {"employee_id": "E1", "final_score": 0.8, "eligible": True},
            ]),
        }
        result = solve_ad_hoc_project(role_scores)
        assert result.empty
        assert list(result.columns) == ["role", "employee_id", "final_score"]

    def test_normal_solve_still_returns_expected_columns(self):
        """Sanity check: the non-empty path's columns match the
        documented empty-path columns exactly, so callers see one
        consistent shape either way."""
        role_scores = {
            "Backend Dev": pd.DataFrame([
                {"employee_id": "E1", "final_score": 0.9, "eligible": True},
            ]),
        }
        result = solve_ad_hoc_project(role_scores)
        assert not result.empty
        assert list(result.columns) == ["role", "employee_id", "final_score"]
        assert result.iloc[0]["employee_id"] == "E1"